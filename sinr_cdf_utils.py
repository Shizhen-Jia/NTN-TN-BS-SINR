from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from BeamformingCalc import nulling_bf_music_noncoh, svd_bf
from ntn_music_detection import (
    build_ntn_truth_from_paths,
    collapse_cir_to_narrowband,
    run_music_standard_pipeline,
    summarize_ntn_music_quality,
)


def _safe_db(power_linear: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.asarray(power_linear, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, float(eps)))
    if np.isscalar(power_linear):
        return float(out)
    return out


def _interference_power_per_rx(h_ntn_tx: np.ndarray, beam: np.ndarray) -> np.ndarray:
    """Per-NTN received interference power for one TX beam.

    Parameters
    ----------
    h_ntn_tx : np.ndarray
        Shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).
    beam : np.ndarray
        Shape (num_tx_ant, 1) or (num_tx_ant,).
    """
    h = np.asarray(h_ntn_tx, dtype=np.complex128)
    v = np.asarray(beam, dtype=np.complex128).reshape(-1, 1)
    if h.ndim != 3:
        raise ValueError("h_ntn_tx must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx_ant).")
    if h.shape[2] != v.shape[0]:
        raise ValueError(
            f"Beam dimension mismatch: h_ntn_tx has {h.shape[2]} TX antennas, beam has {v.shape[0]}."
        )
    eff = np.einsum("nra,ab->nr", h, v, optimize=True)
    return np.sum(np.abs(eff) ** 2, axis=1).real.astype(np.float64)


def pair_tn_to_strongest_tx(
    h_tn_all: np.ndarray,
    *,
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Pair each TN to its strongest valid TX over all BS sectors.

    Pairing rule:
    1. Remove zero channels.
    2. Keep only TXs with direct-signal channel norm > h_tn_th.
    3. Each TN is paired to the strongest remaining TX only.
    """
    h = np.asarray(h_tn_all, dtype=np.complex128)
    if h.ndim != 4:
        raise ValueError("h_tn_all must have shape (num_tn_rx, num_tn_rx_ant, num_tx, num_tx_ant).")

    num_tn_rx, _num_tn_rx_ant, num_tx_total, _num_tx_ant = h.shape
    h_flat = np.transpose(h, (0, 2, 1, 3)).reshape(num_tn_rx, num_tx_total, -1)
    h_norms = np.linalg.norm(h_flat, axis=2)

    nonzero_mask = h_norms > float(eps)
    valid_mask = nonzero_mask & (h_norms > float(h_tn_th))

    best_tx_idx = np.full((num_tn_rx,), -1, dtype=int)
    best_h_norm = np.zeros((num_tn_rx,), dtype=np.float64)
    pairs_by_tx: Dict[int, List[Dict[str, Any]]] = {int(t): [] for t in range(num_tx_total)}
    pairs: List[Dict[str, Any]] = []

    for tn_idx in range(num_tn_rx):
        tx_candidates = np.flatnonzero(valid_mask[tn_idx])
        if tx_candidates.size == 0:
            continue

        tx_local_best = tx_candidates[int(np.argmax(h_norms[tn_idx, tx_candidates]))]
        best_tx_idx[tn_idx] = int(tx_local_best)
        best_h_norm[tn_idx] = float(h_norms[tn_idx, tx_local_best])

        h_tn = np.asarray(h[tn_idx, :, tx_local_best, :], dtype=np.complex128).T
        w_t, w_r = svd_bf(h_tn, tx_antennas)
        snr_raw_linear = (
            np.abs((w_t.conj().T @ h_tn @ w_r).item()) ** 2
            * float(tx_power)
            / float(snr_noise_power)
        )

        pair_record = {
            "tn_idx": int(tn_idx),
            "tx_idx": int(tx_local_best),
            "h_tn": h_tn,
            "h_norm": float(best_h_norm[tn_idx]),
            "w_t": np.asarray(w_t, dtype=np.complex128),
            "w_r": np.asarray(w_r, dtype=np.complex128),
            "snr_raw_db": float(_safe_db(snr_raw_linear, eps=eps)),
        }
        pairs_by_tx[int(tx_local_best)].append(pair_record)
        pairs.append(pair_record)

    pair_counts_by_tx = np.array([len(pairs_by_tx[int(t)]) for t in range(num_tx_total)], dtype=int)
    min_count = int(pair_counts_by_tx.min()) if pair_counts_by_tx.size > 0 else 0

    return {
        "h_norms": h_norms,
        "nonzero_mask": nonzero_mask,
        "valid_mask": valid_mask,
        "best_tx_idx": best_tx_idx,
        "best_h_norm": best_h_norm,
        "pairs": pairs,
        "paired_tn_idx": np.asarray([row["tn_idx"] for row in pairs], dtype=int),
        "pairs_by_tx": pairs_by_tx,
        "pair_counts_by_tx": pair_counts_by_tx,
        "min_count": min_count,
    }


def build_music_tx_lookup(
    ntn_music_out: Dict[str, Any],
    *,
    num_ntn_rx: int,
    num_tx_total: int,
    num_tx_ant: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect detected NTN steering vectors and weights per TX."""
    pair_rx = np.asarray(ntn_music_out.get("pair_rx_idx", []), dtype=int)
    pair_t = np.asarray(ntn_music_out.get("pair_t_idx", []), dtype=int)
    pair_u = np.asarray(ntn_music_out.get("pair_u_hat", []), dtype=np.complex128)
    pair_alpha_hat = np.asarray(ntn_music_out.get("pair_alpha_hat_raw", []), dtype=np.complex128)

    if pair_u.ndim == 1:
        if pair_u.size == 0:
            pair_u = np.empty((0, num_tx_ant), dtype=np.complex128)
        elif pair_u.size == num_tx_ant:
            pair_u = pair_u.reshape(1, -1)
        else:
            raise ValueError(f"Unexpected pair_u shape: {pair_u.shape}")
    if pair_u.ndim != 2:
        raise ValueError(f"pair_u_hat must be 2D after reshape, got {pair_u.shape}")
    if pair_u.shape[1] != num_tx_ant and pair_u.shape[0] > 0:
        raise ValueError(
            f"pair_u_hat antenna dimension mismatch: expected {num_tx_ant}, got {pair_u.shape[1]}."
        )
    if not (pair_rx.size == pair_t.size == pair_alpha_hat.size == pair_u.shape[0]):
        raise ValueError("Inconsistent MUSIC pair lengths in ntn_music_out.")

    lookup: Dict[int, Dict[str, np.ndarray]] = {}
    for tx_idx in range(num_tx_total):
        tx_mask = pair_t == int(tx_idx)
        if not np.any(tx_mask):
            lookup[int(tx_idx)] = {
                "rx_detected": np.empty((0,), dtype=int),
                "u": np.empty((0, num_tx_ant), dtype=np.complex128),
                "g": np.empty((0,), dtype=np.float64),
            }
            continue

        rx_t = pair_rx[tx_mask]
        rx_t = rx_t[(rx_t >= 0) & (rx_t < num_ntn_rx)]
        rx_detected = np.unique(rx_t)

        u_t = np.asarray(pair_u[tx_mask], dtype=np.complex128)
        alpha_t = np.asarray(pair_alpha_hat[tx_mask], dtype=np.complex128)
        finite_u = np.all(np.isfinite(np.real(u_t)) & np.isfinite(np.imag(u_t)), axis=1)
        finite_a = np.isfinite(np.real(alpha_t)) & np.isfinite(np.imag(alpha_t))
        keep = finite_u & finite_a

        if np.any(keep):
            u_keep = u_t[keep]
            g_keep = np.abs(alpha_t[keep]) ** 2
        else:
            u_keep = np.empty((0, num_tx_ant), dtype=np.complex128)
            g_keep = np.empty((0,), dtype=np.float64)

        lookup[int(tx_idx)] = {
            "rx_detected": rx_detected.astype(int),
            "u": np.asarray(u_keep, dtype=np.complex128),
            "g": np.asarray(g_keep, dtype=np.float64),
        }

    return lookup


def run_small_round(
    h_ntn_all: np.ndarray,
    *,
    pairs_by_tx: Dict[int, List[Dict[str, Any]]],
    music_lookup: Dict[int, Dict[str, np.ndarray]],
    round_idx: int,
    lambda_ranges: Iterable[float],
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Run one small simulation round where every TX serves one paired TN."""
    h_ntn = np.asarray(h_ntn_all, dtype=np.complex128)
    if h_ntn.ndim != 4:
        raise ValueError("h_ntn_all must have shape (num_ntn_rx, num_ntn_rx_ant, num_tx, num_tx_ant).")

    num_ntn_rx = int(h_ntn.shape[0])
    num_tx_total = int(h_ntn.shape[2])
    lambda_list = [float(v) for v in lambda_ranges]

    raw_snr_db: List[float] = []
    null_snr_db: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    raw_inr_power = np.zeros((num_ntn_rx,), dtype=np.float64)
    null_inr_power = {lambda_: np.zeros((num_ntn_rx,), dtype=np.float64) for lambda_ in lambda_list}
    detected_mask = np.zeros((num_ntn_rx,), dtype=bool)

    for tx_idx in range(num_tx_total):
        tx_pairs = pairs_by_tx.get(int(tx_idx), [])
        if round_idx >= len(tx_pairs):
            raise ValueError(
                f"round_idx={round_idx} exceeds paired TN count for tx_idx={tx_idx}. "
                "Check min_count scheduling."
            )

        pair = tx_pairs[round_idx]
        h_tn = np.asarray(pair["h_tn"], dtype=np.complex128)
        w_t = np.asarray(pair["w_t"], dtype=np.complex128)
        w_r = np.asarray(pair["w_r"], dtype=np.complex128)
        raw_snr_db.append(float(pair["snr_raw_db"]))

        h_ntn_tx = np.asarray(h_ntn[:, :, tx_idx, :], dtype=np.complex128)
        raw_inr_power += _interference_power_per_rx(h_ntn_tx, w_t)

        lookup = music_lookup.get(int(tx_idx), {})
        rx_detected = np.asarray(lookup.get("rx_detected", np.empty((0,), dtype=int)), dtype=int)
        if rx_detected.size > 0:
            detected_mask[rx_detected] = True

        u_t = np.asarray(lookup.get("u", np.empty((0, h_tn.shape[0]), dtype=np.complex128)), dtype=np.complex128)
        g_t = np.asarray(lookup.get("g", np.empty((0,), dtype=np.float64)), dtype=np.float64)

        for lambda_ in lambda_list:
            v_null, _, _, _ = nulling_bf_music_noncoh(h_tn, w_r, u_t, g_t, lambda_, eps=eps)
            null_snr_linear = (
                np.abs((v_null.conj().T @ h_tn @ w_r).item()) ** 2
                * float(tx_power)
                / float(snr_noise_power)
            )
            null_snr_db[lambda_].append(float(_safe_db(null_snr_linear, eps=eps)))
            null_inr_power[lambda_] += _interference_power_per_rx(h_ntn_tx, v_null)

    raw_inr_db = (
        _safe_db(raw_inr_power[detected_mask] * float(tx_power) / float(inr_noise_power), eps=eps)
        if np.any(detected_mask)
        else np.empty((0,), dtype=np.float64)
    )
    null_inr_db = {
        lambda_: (
            _safe_db(
                null_inr_power[lambda_][detected_mask] * float(tx_power) / float(inr_noise_power),
                eps=eps,
            )
            if np.any(detected_mask)
            else np.empty((0,), dtype=np.float64)
        )
        for lambda_ in lambda_list
    }

    return {
        "raw_snr_db": np.asarray(raw_snr_db, dtype=np.float64),
        "null_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in null_snr_db.items()},
        "raw_inr_db": np.asarray(raw_inr_db, dtype=np.float64),
        "null_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in null_inr_db.items()},
        "detected_mask": detected_mask,
        "detected_count": int(np.count_nonzero(detected_mask)),
    }


def run_sinr_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    lambda_ranges: Iterable[float],
    h_tn_th: float,
    tx_antennas: int,
    tx_power: float,
    snr_noise_power: float,
    inr_noise_power: float,
    music_kwargs: Dict[str, Any],
    sionna_phi_is_global: bool = True,
    theta_display_mode: str = "elevation",
    eps: float = 1e-12,
    plot_first_sim_only: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run n macro simulations and m min-count small rounds per macro simulation."""
    if int(num_macro_sims) <= 0:
        raise ValueError("num_macro_sims must be positive.")

    try:
        from tqdm.auto import trange
    except Exception:
        trange = None

    lambda_list = [float(v) for v in lambda_ranges]
    raw_snr_all: List[float] = []
    raw_inr_all: List[float] = []
    null_snr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    null_inr_all: Dict[float, List[float]] = {lambda_: [] for lambda_ in lambda_list}
    macro_stats: List[Dict[str, Any]] = []
    bs_pos_ref: np.ndarray | None = None

    iterator = (
        trange(int(num_macro_sims), desc="Monte Carlo", leave=False)
        if show_progress and trange is not None
        else range(int(num_macro_sims))
    )

    for sim_idx in iterator:
        pos_kwargs = dict(compute_positions_kwargs)
        if plot_first_sim_only and sim_idx > 0:
            for key in ("plot_grid", "plot_bs", "plot_tn", "plot_ntn"):
                if key in pos_kwargs:
                    pos_kwargs[key] = False

        scene_config.compute_positions(**pos_kwargs)
        tx_pos = np.asarray(scene_config.tx_pos, dtype=np.float64)
        if bs_pos_ref is None:
            bs_pos_ref = tx_pos.copy()
        elif tx_pos.shape != bs_pos_ref.shape or not np.allclose(tx_pos, bs_pos_ref):
            raise RuntimeError(
                "BS positions changed across macro simulations. "
                "The requested experiment assumes fixed BS positions."
            )

        scene_config.compute_paths(**compute_paths_kwargs)
        h_tn_all = collapse_cir_to_narrowband(scene_config.a_tn)
        h_ntn_all = collapse_cir_to_narrowband(scene_config.a_ntn)

        pairing = pair_tn_to_strongest_tx(
            h_tn_all,
            h_tn_th=float(h_tn_th),
            tx_antennas=int(tx_antennas),
            tx_power=float(tx_power),
            snr_noise_power=float(snr_noise_power),
            eps=eps,
        )
        min_count = int(pairing["min_count"])
        num_ntn_rx = int(h_ntn_all.shape[0])
        num_tx_total = int(h_ntn_all.shape[2])
        num_tx_ant = int(h_ntn_all.shape[3])

        ntn_music_out = run_music_standard_pipeline(
            h_ntn_all,
            **music_kwargs,
        )
        ntn_truth = build_ntn_truth_from_paths(
            scene_config.paths_ntn,
            scene_config.a_ntn,
            num_tx_total=num_tx_total,
            nsect=int(music_kwargs["nsect"]),
            sionna_phi_is_global=bool(sionna_phi_is_global),
        )
        music_quality = summarize_ntn_music_quality(
            h_ntn_all,
            ntn_music_out,
            ntn_truth["pair_map"],
            theta_display_mode=str(theta_display_mode),
            eps=eps,
        )
        music_lookup = build_music_tx_lookup(
            ntn_music_out,
            num_ntn_rx=num_ntn_rx,
            num_tx_total=num_tx_total,
            num_tx_ant=num_tx_ant,
        )

        detected_rx_union = np.asarray(ntn_music_out.get("detected_rx_indices_unique", []), dtype=int)
        interfered_ntn_count = int(np.count_nonzero(np.any(np.abs(h_ntn_all) > eps, axis=(1, 2, 3))))
        pair_counts_by_tx = np.asarray(pairing["pair_counts_by_tx"], dtype=int)

        macro_stats.append(
            {
                "sim_idx": int(sim_idx),
                "min_count": int(min_count),
                "pair_counts_by_tx": pair_counts_by_tx.copy(),
                "detected_ntn_count": int(detected_rx_union.size),
                "interfered_ntn_count": interfered_ntn_count,
                "angle_metrics": music_quality["angle_metrics"],
                "detected_subset_metrics": music_quality["detected_subset_metrics"],
                "detected_pairs_summary": music_quality["detected_pairs_summary"],
            }
        )

        if min_count <= 0:
            continue

        for round_idx in range(min_count):
            round_out = run_small_round(
                h_ntn_all,
                pairs_by_tx=pairing["pairs_by_tx"],
                music_lookup=music_lookup,
                round_idx=int(round_idx),
                lambda_ranges=lambda_list,
                tx_power=float(tx_power),
                snr_noise_power=float(snr_noise_power),
                inr_noise_power=float(inr_noise_power),
                eps=eps,
            )

            raw_snr_all.extend(np.asarray(round_out["raw_snr_db"], dtype=np.float64).tolist())
            raw_inr_all.extend(np.asarray(round_out["raw_inr_db"], dtype=np.float64).tolist())
            for lambda_ in lambda_list:
                null_snr_all[lambda_].extend(
                    np.asarray(round_out["null_snr_db"][lambda_], dtype=np.float64).tolist()
                )
                null_inr_all[lambda_].extend(
                    np.asarray(round_out["null_inr_db"][lambda_], dtype=np.float64).tolist()
                )

    return {
        "raw_snr_db": np.asarray(raw_snr_all, dtype=np.float64),
        "raw_inr_db": np.asarray(raw_inr_all, dtype=np.float64),
        "null_snr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in null_snr_all.items()},
        "null_inr_db": {lambda_: np.asarray(vals, dtype=np.float64) for lambda_, vals in null_inr_all.items()},
        "macro_stats": macro_stats,
        "bs_pos_ref": bs_pos_ref,
        "lambda_ranges": np.asarray(lambda_list, dtype=np.float64),
    }


def save_experiment_metrics(
    experiment_out: Dict[str, Any],
    *,
    result_dir: str | Path = "result",
    output_name: str = "sinr_cdf_metrics.npz",
) -> Path:
    """Save experiment arrays and macro statistics for later reuse."""
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / output_name

    save_dict: Dict[str, Any] = {
        "raw_snr_db": np.asarray(experiment_out["raw_snr_db"], dtype=np.float64),
        "raw_inr_db": np.asarray(experiment_out["raw_inr_db"], dtype=np.float64),
        "lambda_ranges": np.asarray(experiment_out["lambda_ranges"], dtype=np.float64),
    }
    for lambda_, vals in experiment_out["null_snr_db"].items():
        save_dict[f"null_snr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)
    for lambda_, vals in experiment_out["null_inr_db"].items():
        save_dict[f"null_inr_db_{lambda_:.0e}"] = np.asarray(vals, dtype=np.float64)

    macro_stats = experiment_out.get("macro_stats", [])
    save_dict["macro_stats_sim_idx"] = np.asarray([row["sim_idx"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_min_count"] = np.asarray([row["min_count"] for row in macro_stats], dtype=int)
    save_dict["macro_stats_detected_ntn_count"] = np.asarray(
        [row["detected_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    save_dict["macro_stats_interfered_ntn_count"] = np.asarray(
        [row["interfered_ntn_count"] for row in macro_stats],
        dtype=int,
    )
    if macro_stats:
        save_dict["macro_stats_pair_counts_by_tx"] = np.stack(
            [np.asarray(row["pair_counts_by_tx"], dtype=int) for row in macro_stats],
            axis=0,
        )
        save_dict["macro_stats_phi_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("phi_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_elev_mae_deg"] = np.asarray(
            [row.get("angle_metrics", {}).get("elev_mae_deg", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_angle_match_count"] = np.asarray(
            [row.get("angle_metrics", {}).get("matched_pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_detected_subset_count"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("count", 0) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_nrmse"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("nrmse", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_cos_sim"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("cos_sim", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_mag_mae"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("mag_mae", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_subset_power_ratio_db"] = np.asarray(
            [row.get("detected_subset_metrics", {}).get("power_ratio_db", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_count"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("pairs", 0) for row in macro_stats],
            dtype=int,
        )
        save_dict["macro_stats_detected_pairs_nrmse_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_nrmse_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("nrmse_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_mean"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_mean", np.nan) for row in macro_stats],
            dtype=np.float64,
        )
        save_dict["macro_stats_detected_pairs_cos_median"] = np.asarray(
            [row.get("detected_pairs_summary", {}).get("cos_median", np.nan) for row in macro_stats],
            dtype=np.float64,
        )

    np.savez(save_path, **save_dict)
    return save_path


save_sinr_cdf_metrics = save_experiment_metrics


def _beamformed_link_power_linear(
    h_link: np.ndarray,
    w_tx: np.ndarray,
    w_rx: np.ndarray,
) -> float:
    """Beamformed link power |w_tx^H H w_rx|^2 for H shaped (num_tx_ant, num_rx_ant)."""
    h = np.asarray(h_link, dtype=np.complex128)
    wt = np.asarray(w_tx, dtype=np.complex128).reshape(-1, 1)
    wr = np.asarray(w_rx, dtype=np.complex128).reshape(-1, 1)
    if h.ndim != 2:
        raise ValueError("h_link must be 2D with shape (num_tx_ant, num_rx_ant).")
    if h.shape != (wt.shape[0], wr.shape[0]):
        raise ValueError(
            "Beam/channel dimension mismatch: "
            f"h_link has shape {h.shape}, w_tx has {wt.shape[0]} entries, "
            f"w_rx has {wr.shape[0]} entries."
        )
    return float(np.abs((wt.conj().T @ h @ wr).item()) ** 2)


def _rx_slice_to_link_stack(h_slice: np.ndarray) -> np.ndarray:
    """Convert a fixed-RX tensor slice to link matrices.

    Input shape:
        (num_rx_ant, num_tx_nodes, num_tx_ant)
    Output shape:
        (num_tx_nodes, num_tx_ant, num_rx_ant)
    """
    h = np.asarray(h_slice, dtype=np.complex128)
    if h.ndim != 3:
        raise ValueError("h_slice must have shape (num_rx_ant, num_tx_nodes, num_tx_ant).")
    return np.transpose(h, (1, 2, 0))


def _aggregate_interference_power_linear(
    h_links: np.ndarray,
    *,
    w_rx: np.ndarray,
    w_tx: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Sum interference power over many transmitters after RX combining.

    Parameters
    ----------
    h_links : np.ndarray
        Shape (num_interferers, num_tx_ant, num_rx_ant).
    w_rx : np.ndarray
        Receive combiner, shape (num_rx_ant, 1) or (num_rx_ant,).
    w_tx : np.ndarray | None
        Common transmit beam used by every interferer. If None, an equal-weight
        beam is used across each interferer's TX antennas.
    """
    h = np.asarray(h_links, dtype=np.complex128)
    if h.size == 0:
        return 0.0
    if h.ndim != 3:
        raise ValueError("h_links must have shape (num_interferers, num_tx_ant, num_rx_ant).")

    wr = np.asarray(w_rx, dtype=np.complex128).reshape(-1, 1)
    if h.shape[2] != wr.shape[0]:
        raise ValueError(
            f"Receive-combiner mismatch: h_links has {h.shape[2]} RX antennas, w_rx has {wr.shape[0]}."
        )

    num_tx_ant = int(h.shape[1])
    if w_tx is None:
        wt = np.ones((num_tx_ant, 1), dtype=np.complex128) / np.sqrt(max(num_tx_ant, 1))
    else:
        wt = np.asarray(w_tx, dtype=np.complex128).reshape(-1, 1)
        if wt.shape[0] != num_tx_ant:
            raise ValueError(
                f"Transmit-beam mismatch: h_links has {num_tx_ant} TX antennas, w_tx has {wt.shape[0]}."
            )

    eff = np.einsum(
        "a,kab,b->k",
        np.conjugate(wt[:, 0]),
        h,
        wr[:, 0],
        optimize=True,
    )
    power = np.sum(np.abs(eff) ** 2).real
    return float(max(power, 0.0))


def compute_two_mode_sinr_samples(
    h_bs_to_tn: np.ndarray,
    h_tn_to_bs: np.ndarray,
    h_ntn_to_tn: np.ndarray | None,
    h_ntn_to_bs: np.ndarray | None,
    *,
    pairs: List[Dict[str, Any]],
    bs_tx_power: float,
    tn_tx_power: float,
    ntn_tx_power: float,
    tn_noise_power: float,
    bs_noise_power: float,
    signal_threshold: float,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Compute pair-wise SINR samples for the two requested experiment modes.

    NTN nodes are treated purely as interferers. No MUSIC detection is used.
    A SINR sample is recorded for a mode only if that mode's direct-signal
    channel norm exceeds `signal_threshold`.
    """
    h_dl_all = np.asarray(h_bs_to_tn, dtype=np.complex128)
    h_ul_all = np.asarray(h_tn_to_bs, dtype=np.complex128)
    if h_dl_all.ndim != 4:
        raise ValueError("h_bs_to_tn must have shape (num_tn, num_tn_rx_ant, num_bs_sector, num_bs_tx_ant).")
    if h_ul_all.ndim != 4:
        raise ValueError("h_tn_to_bs must have shape (num_bs_sector, num_bs_rx_ant, num_tn, num_tn_tx_ant).")

    h_ntn_tn_all = None if h_ntn_to_tn is None else np.asarray(h_ntn_to_tn, dtype=np.complex128)
    h_ntn_bs_all = None if h_ntn_to_bs is None else np.asarray(h_ntn_to_bs, dtype=np.complex128)

    mode1_sinr_db: List[float] = []
    mode2_sinr_db: List[float] = []

    for pair in pairs:
        tn_idx = int(pair["tn_idx"])
        tx_idx = int(pair["tx_idx"])

        # Mode 1: BS -> TN signal at TN, NTN -> TN interference.
        h_dl = np.asarray(pair["h_tn"], dtype=np.complex128)
        h_dl_norm = float(pair.get("h_norm", np.linalg.norm(h_dl)))
        if h_dl.ndim == 2 and h_dl_norm > max(float(signal_threshold), float(eps)):
            w_bs_tx = np.asarray(pair["w_t"], dtype=np.complex128)
            w_tn_rx = np.asarray(pair["w_r"], dtype=np.complex128)
            sig_mode1 = _beamformed_link_power_linear(h_dl, w_bs_tx, w_tn_rx) * float(bs_tx_power)

            if h_ntn_tn_all is None or h_ntn_tn_all.size == 0:
                int_mode1 = 0.0
            else:
                h_ntn_to_this_tn = _rx_slice_to_link_stack(h_ntn_tn_all[tn_idx, :, :, :])
                int_mode1 = (
                    _aggregate_interference_power_linear(
                        h_ntn_to_this_tn,
                        w_rx=w_tn_rx,
                        eps=eps,
                    )
                    * float(ntn_tx_power)
                )

            sinr_mode1 = sig_mode1 / max(int_mode1 + float(tn_noise_power), float(eps))
            mode1_sinr_db.append(float(_safe_db(sinr_mode1, eps=eps)))

        # Mode 2: TN -> BS signal at BS, NTN -> BS interference.
        h_ul = np.asarray(h_ul_all[tx_idx, :, tn_idx, :], dtype=np.complex128).T
        h_ul_norm = float(np.linalg.norm(h_ul))
        if h_ul.ndim == 2 and h_ul_norm > max(float(signal_threshold), float(eps)):
            w_tn_tx, w_bs_rx = svd_bf(h_ul, h_ul.shape[0])
            sig_mode2 = _beamformed_link_power_linear(h_ul, w_tn_tx, w_bs_rx) * float(tn_tx_power)

            if h_ntn_bs_all is None or h_ntn_bs_all.size == 0:
                int_mode2 = 0.0
            else:
                h_ntn_to_this_bs = _rx_slice_to_link_stack(h_ntn_bs_all[tx_idx, :, :, :])
                int_mode2 = (
                    _aggregate_interference_power_linear(
                        h_ntn_to_this_bs,
                        w_rx=w_bs_rx,
                        eps=eps,
                    )
                    * float(ntn_tx_power)
                )

            sinr_mode2 = sig_mode2 / max(int_mode2 + float(bs_noise_power), float(eps))
            mode2_sinr_db.append(float(_safe_db(sinr_mode2, eps=eps)))

    return {
        "mode1_sinr_db": np.asarray(mode1_sinr_db, dtype=np.float64),
        "mode2_sinr_db": np.asarray(mode2_sinr_db, dtype=np.float64),
    }


def _build_sinr_channel_kwargs(compute_paths_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    tx_power_dbm_default = float(compute_paths_kwargs.get("tx_power_dbm", 30.0))
    return {
        "tx_rows": int(compute_paths_kwargs.get("tx_rows", 8)),
        "tx_cols": int(compute_paths_kwargs.get("tx_cols", 8)),
        "tn_rows": int(compute_paths_kwargs.get("tn_rx_rows", 1)),
        "tn_cols": int(compute_paths_kwargs.get("tn_rx_cols", 1)),
        "max_depth": int(compute_paths_kwargs.get("max_depth", 3)),
        "bandwidth": float(compute_paths_kwargs.get("bandwidth", 100e6)),
        "tn_tx_power_dbm": float(compute_paths_kwargs.get("tn_tx_power_dbm", tx_power_dbm_default)),
        "ntn_tx_power_dbm": float(compute_paths_kwargs.get("ntn_tx_power_dbm", tx_power_dbm_default)),
    }


def run_two_mode_sinr_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    ntn_drop_counts: Iterable[int],
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    h_tn_th: float,
    bs_tx_power: float,
    tn_tx_power: float,
    ntn_tx_power: float,
    tn_noise_power: float,
    bs_noise_power: float,
    eps: float = 1e-12,
    plot_first_sim_only: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the two requested SINR modes over multiple NTN drop counts."""
    if int(num_macro_sims) <= 0:
        raise ValueError("num_macro_sims must be positive.")

    try:
        from tqdm.auto import trange
    except Exception:
        trange = None

    ntn_counts = [int(v) for v in ntn_drop_counts]
    if not ntn_counts:
        raise ValueError("ntn_drop_counts must contain at least one NTN count.")

    sinr_channel_kwargs = _build_sinr_channel_kwargs(compute_paths_kwargs)
    mode1_sinr_all: Dict[int, List[float]] = {count: [] for count in ntn_counts}
    mode2_sinr_all: Dict[int, List[float]] = {count: [] for count in ntn_counts}
    macro_stats: Dict[int, List[Dict[str, Any]]] = {count: [] for count in ntn_counts}
    bs_pos_ref: np.ndarray | None = None
    did_plot_layout = False

    for ntn_count in ntn_counts:
        iterator = (
            trange(int(num_macro_sims), desc=f"NTN={ntn_count}", leave=False)
            if show_progress and trange is not None
            else range(int(num_macro_sims))
        )

        for sim_idx in iterator:
            pos_kwargs = dict(compute_positions_kwargs)
            pos_kwargs["ntn_rx"] = int(ntn_count)
            if plot_first_sim_only and did_plot_layout:
                for key in ("plot_grid", "plot_bs", "plot_tn", "plot_ntn"):
                    if key in pos_kwargs:
                        pos_kwargs[key] = False

            scene_config.compute_positions(**pos_kwargs)
            did_plot_layout = did_plot_layout or any(
                bool(pos_kwargs.get(k, False)) for k in ("plot_grid", "plot_bs", "plot_tn", "plot_ntn")
            )

            tx_pos = np.asarray(scene_config.tx_pos, dtype=np.float64)
            if bs_pos_ref is None:
                bs_pos_ref = tx_pos.copy()
            elif tx_pos.shape != bs_pos_ref.shape or not np.allclose(tx_pos, bs_pos_ref):
                raise RuntimeError(
                    "BS positions changed across simulations. "
                    "The requested experiment assumes fixed BS positions."
                )

            scene_config.compute_paths(**compute_paths_kwargs)
            scene_config.compute_sinr_channels(**sinr_channel_kwargs)

            h_bs_to_tn = collapse_cir_to_narrowband(scene_config.a_tn)
            h_tn_to_bs = collapse_cir_to_narrowband(scene_config.a_tn_to_bs)
            h_ntn_to_tn = (
                None
                if scene_config.a_ntn_to_tn is None
                else collapse_cir_to_narrowband(scene_config.a_ntn_to_tn)
            )
            h_ntn_to_bs = (
                None
                if scene_config.a_ntn_to_bs is None
                else collapse_cir_to_narrowband(scene_config.a_ntn_to_bs)
            )

            pairing = pair_tn_to_strongest_tx(
                h_bs_to_tn,
                h_tn_th=float(h_tn_th),
                tx_antennas=int(h_bs_to_tn.shape[3]),
                tx_power=float(bs_tx_power),
                snr_noise_power=float(tn_noise_power),
                eps=eps,
            )

            sinr_out = compute_two_mode_sinr_samples(
                h_bs_to_tn,
                h_tn_to_bs,
                h_ntn_to_tn,
                h_ntn_to_bs,
                pairs=pairing["pairs"],
                bs_tx_power=float(bs_tx_power),
                tn_tx_power=float(tn_tx_power),
                ntn_tx_power=float(ntn_tx_power),
                tn_noise_power=float(tn_noise_power),
                bs_noise_power=float(bs_noise_power),
                signal_threshold=float(h_tn_th),
                eps=eps,
            )

            mode1_sinr_all[int(ntn_count)].extend(sinr_out["mode1_sinr_db"].tolist())
            mode2_sinr_all[int(ntn_count)].extend(sinr_out["mode2_sinr_db"].tolist())

            pair_counts_by_tx = np.asarray(pairing["pair_counts_by_tx"], dtype=int)
            macro_stats[int(ntn_count)].append(
                {
                    "sim_idx": int(sim_idx),
                    "ntn_count": int(ntn_count),
                    "paired_tn_count": int(len(pairing["pairs"])),
                    "num_tx_with_pairs": int(np.count_nonzero(pair_counts_by_tx > 0)),
                    "empty_tx_count": int(np.count_nonzero(pair_counts_by_tx == 0)),
                    "pair_counts_by_tx": pair_counts_by_tx.copy(),
                    "mode1_samples": int(sinr_out["mode1_sinr_db"].size),
                    "mode2_samples": int(sinr_out["mode2_sinr_db"].size),
                }
            )

    return {
        "ntn_drop_counts": np.asarray(ntn_counts, dtype=int),
        "mode1_sinr_db": {
            int(count): np.asarray(vals, dtype=np.float64) for count, vals in mode1_sinr_all.items()
        },
        "mode2_sinr_db": {
            int(count): np.asarray(vals, dtype=np.float64) for count, vals in mode2_sinr_all.items()
        },
        "macro_stats": macro_stats,
        "bs_pos_ref": bs_pos_ref,
    }


def save_two_mode_sinr_metrics(
    experiment_out: Dict[str, Any],
    *,
    result_dir: str | Path = "result",
    output_name: str = "two_mode_sinr_metrics.npz",
) -> Path:
    """Save the new two-mode SINR experiment outputs to an NPZ file."""
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    save_path = result_path / output_name

    save_dict: Dict[str, Any] = {
        "ntn_drop_counts": np.asarray(experiment_out["ntn_drop_counts"], dtype=int),
    }

    for count, vals in experiment_out.get("mode1_sinr_db", {}).items():
        save_dict[f"mode1_sinr_db_ntn_{int(count)}"] = np.asarray(vals, dtype=np.float64)
    for count, vals in experiment_out.get("mode2_sinr_db", {}).items():
        save_dict[f"mode2_sinr_db_ntn_{int(count)}"] = np.asarray(vals, dtype=np.float64)

    macro_stats = experiment_out.get("macro_stats", {})
    for count, rows in macro_stats.items():
        prefix = f"macro_stats_ntn_{int(count)}"
        save_dict[f"{prefix}_sim_idx"] = np.asarray([row["sim_idx"] for row in rows], dtype=int)
        save_dict[f"{prefix}_paired_tn_count"] = np.asarray([row["paired_tn_count"] for row in rows], dtype=int)
        save_dict[f"{prefix}_num_tx_with_pairs"] = np.asarray([row["num_tx_with_pairs"] for row in rows], dtype=int)
        save_dict[f"{prefix}_empty_tx_count"] = np.asarray([row["empty_tx_count"] for row in rows], dtype=int)
        save_dict[f"{prefix}_mode1_samples"] = np.asarray([row["mode1_samples"] for row in rows], dtype=int)
        save_dict[f"{prefix}_mode2_samples"] = np.asarray([row["mode2_samples"] for row in rows], dtype=int)
        if rows:
            save_dict[f"{prefix}_pair_counts_by_tx"] = np.stack(
                [np.asarray(row["pair_counts_by_tx"], dtype=int) for row in rows],
                axis=0,
            )

    np.savez(save_path, **save_dict)
    return save_path


def _cdf_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    vals_sorted = np.sort(vals)
    cdf = np.arange(1, vals_sorted.size + 1, dtype=np.float64) / float(vals_sorted.size)
    return vals_sorted, cdf


def plot_two_mode_sinr_cdfs(
    experiment_out: Dict[str, Any],
    *,
    result_dir: str | Path = "result",
    output_prefix: str = "two_mode_sinr_cdf",
    figure_size: Tuple[float, float] = (3.5, 2.6),
) -> Dict[str, Dict[str, Path]]:
    """Plot one CDF figure per mode with one curve per NTN drop count."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.4,
        }
    )

    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8c564b", "#17becf"]
    styles = ["-", "--", "-.", ":", (0, (4, 1, 1, 1)), (0, (6, 2))]

    mode_specs = [
        (
            "mode1_sinr_db",
            "mode1",
            "Mode 1 SINR CDF",
            "Signal: BS, Interference: NTN",
        ),
        (
            "mode2_sinr_db",
            "mode2",
            "Mode 2 SINR CDF",
            "Signal: TN, Interference: NTN",
        ),
    ]

    out_paths: Dict[str, Dict[str, Path]] = {}
    for key, tag, title, subtitle in mode_specs:
        fig, ax = plt.subplots(figsize=figure_size)
        data = experiment_out.get(key, {})
        plotted = False
        for idx, ntn_count in enumerate(sorted(int(v) for v in data.keys())):
            x, y = _cdf_curve(data[int(ntn_count)])
            if x.size == 0:
                continue
            ax.plot(
                x,
                y,
                color=colors[idx % len(colors)],
                linestyle=styles[idx % len(styles)],
                label=f"NTN={int(ntn_count)}",
            )
            plotted = True

        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
        if plotted:
            ax.legend(loc="lower right", frameon=True)
        fig.tight_layout(pad=0.2)

        png_path = result_path / f"{output_prefix}_{tag}.png"
        pdf_path = result_path / f"{output_prefix}_{tag}.pdf"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        out_paths[tag] = {"png": png_path, "pdf": pdf_path}

    return out_paths


def run_and_plot_two_mode_sinr_cdf_experiment(
    scene_config: Any,
    *,
    num_macro_sims: int,
    ntn_drop_counts: Iterable[int],
    compute_positions_kwargs: Dict[str, Any],
    compute_paths_kwargs: Dict[str, Any],
    h_tn_th: float,
    bs_tx_power: float,
    tn_tx_power: float,
    ntn_tx_power: float,
    tn_noise_power: float,
    bs_noise_power: float,
    result_dir: str | Path = "result",
    metrics_output_name: str = "two_mode_sinr_metrics.npz",
    figure_output_prefix: str = "two_mode_sinr_cdf",
    eps: float = 1e-12,
    plot_first_sim_only: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper to run, save, and plot the two-mode SINR experiment."""
    experiment_out = run_two_mode_sinr_cdf_experiment(
        scene_config,
        num_macro_sims=num_macro_sims,
        ntn_drop_counts=ntn_drop_counts,
        compute_positions_kwargs=compute_positions_kwargs,
        compute_paths_kwargs=compute_paths_kwargs,
        h_tn_th=h_tn_th,
        bs_tx_power=bs_tx_power,
        tn_tx_power=tn_tx_power,
        ntn_tx_power=ntn_tx_power,
        tn_noise_power=tn_noise_power,
        bs_noise_power=bs_noise_power,
        eps=eps,
        plot_first_sim_only=plot_first_sim_only,
        show_progress=show_progress,
    )
    metrics_path = save_two_mode_sinr_metrics(
        experiment_out,
        result_dir=result_dir,
        output_name=metrics_output_name,
    )
    figure_paths = plot_two_mode_sinr_cdfs(
        experiment_out,
        result_dir=result_dir,
        output_prefix=figure_output_prefix,
    )
    experiment_out["metrics_path"] = metrics_path
    experiment_out["figure_paths"] = figure_paths
    return experiment_out
