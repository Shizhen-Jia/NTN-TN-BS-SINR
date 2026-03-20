from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from BeamformingCalc import svd_bf



def collapse_cir_to_narrowband(cir: np.ndarray) -> np.ndarray:
    """Collapse CIR to narrowband channel tensor with stable axis order.

    Expected CIR axis order from Sionna:
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    The function sums over all trailing axes after tx-ant, returning:
        h_all.shape == (num_rx, num_rx_ant, num_tx, num_tx_ant)
    """
    h = np.asarray(cir, dtype=np.complex128)
    if h.ndim < 4:
        raise ValueError(
            "cir must have at least 4 dims: "
            "[num_rx, num_rx_ant, num_tx, num_tx_ant, ...]."
        )
    if h.ndim == 4:
        return h
    sum_axes = tuple(range(4, h.ndim))
    return np.sum(h, axis=sum_axes)


def _safe_db(power_linear: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.asarray(power_linear, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, float(eps)))
    if np.isscalar(power_linear):
        return float(out)
    return out



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
        "ntn_tx_batch_size": int(compute_paths_kwargs.get("ntn_tx_batch_size", 32)),
    }


def _filter_callable_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keyword arguments accepted by ``func``.

    This lets the notebook use one high-level configuration dict while the
    runner safely forwards only the parameters each scene method supports.
    """
    signature = inspect.signature(func)
    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


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

    path_kwargs = _filter_callable_kwargs(scene_config.compute_paths, compute_paths_kwargs)
    if "compute_ntn_paths" in inspect.signature(scene_config.compute_paths).parameters:
        path_kwargs.setdefault("compute_ntn_paths", False)
    sinr_channel_kwargs = _filter_callable_kwargs(
        scene_config.compute_sinr_channels,
        _build_sinr_channel_kwargs(compute_paths_kwargs),
    )
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

            scene_config.compute_paths(**path_kwargs)
            for attr in ("paths_tn", "paths_ntn"):
                if hasattr(scene_config, attr):
                    setattr(scene_config, attr, None)
            if hasattr(scene_config, "_best_effort_rt_memory_cleanup"):
                scene_config._best_effort_rt_memory_cleanup()
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
            for attr in (
                "a_tn",
                "tau_tn",
                "a_ntn",
                "tau_ntn",
                "a_tn_to_bs",
                "tau_tn_to_bs",
                "a_ntn_to_tn",
                "tau_ntn_to_tn",
                "a_ntn_to_bs",
                "tau_ntn_to_bs",
                "paths_tn_to_bs",
                "paths_ntn_to_tn",
                "paths_ntn_to_bs",
            ):
                if hasattr(scene_config, attr):
                    setattr(scene_config, attr, None)
            if hasattr(scene_config, "_best_effort_rt_memory_cleanup"):
                scene_config._best_effort_rt_memory_cleanup()

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
