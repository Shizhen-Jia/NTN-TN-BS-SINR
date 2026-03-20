# NTN-TN-BS-SINR

Sionna-based simulation code for joint NTN/TN/BS geometry generation, channel extraction, and SINR CDF experiments.

This repository currently contains two kinds of workflows:

- A legacy nulling/MUSIC exploration flow kept in the Python utilities for reference.
- A newer two-mode SINR experiment flow where NTN nodes are treated as pure interferers.

## Recent Updates

- Renamed the main scene module from `SceneConfigSionna2.py` to `SceneConfigSionna.py`.
- Renamed the main CDF notebook from `Nulling_CDF.ipynb` to `SINR_CDF.ipynb`.
- Renamed the main experiment utility module from `nulling_cdf_utils.py` to `sinr_cdf_utils.py`.
- Added extra channel computation support for:
  - `TN -> BS sectors`
  - `NTN -> BS sectors`
  - `NTN -> TN`
- Added a two-mode SINR experiment flow with pair-wise traversal instead of the older `min_count` synchronized sector loop.

## Two-Mode SINR Definition

The current recommended experiment is the two-mode SINR setup:

1. `Mode 1`
   Signal is from `BS -> TN`.
   Interference is from `NTN -> TN`.
   SINR is evaluated at the TN receiver.

2. `Mode 2`
   Signal is from `TN -> BS sector`.
   Interference is from `NTN -> BS sector`.
   SINR is evaluated at the BS receiver.

Important assumptions in the current two-mode pipeline:

- BS positions, TN positions, and NTN positions follow the existing drop logic in `SceneConfigSionna.py`.
- NTN users are treated only as interferers in the two-mode SINR flow.
- No MUSIC detection is used for NTN interferers in the two-mode SINR calculation.
- A SINR sample is recorded only when the corresponding direct-signal channel norm is above the configured threshold.
- TN users are first paired to the strongest valid BS sector, then SINR is evaluated pair by pair.

## Main Files

- [SceneConfigSionna.py](./SceneConfigSionna.py)
  Scene construction, drop generation, BS sector placement, channel generation, and the extra SINR channel computation.

- [sinr_cdf_utils.py](./sinr_cdf_utils.py)
  Pairing, SINR sample generation, experiment runners, metrics saving, and CDF plotting.

- [SINR_CDF.ipynb](./SINR_CDF.ipynb)
  Notebook for experiment setup and plotting.

- [BeamformingCalc.py](./BeamformingCalc.py)
  SVD beamforming and MUSIC-guided nulling helpers.

- [ntn_music_detection.py](./ntn_music_detection.py)
  MUSIC detection and angle-estimation helpers used by the older nulling-oriented flow.

## Recommended Entry Points

For the current two-mode SINR workflow, the main Python APIs are:

- `SceneConfigSionna.compute_positions(...)`
- `SceneConfigSionna.build_standard_arrays(...)`
- `SceneConfigSionna.compute_cir(...)`
- `SceneConfigSionna.compute_two_mode_cirs(...)`
- `sinr_cdf_utils.run_two_mode_sinr_cdf_experiment(...)`
- `sinr_cdf_utils.save_two_mode_sinr_metrics(...)`

Key functions:

- `pair_tn_to_strongest_tx`
- `compute_two_mode_sinr_samples`
- `run_two_mode_sinr_cdf_experiment`
- `save_two_mode_sinr_metrics`

## Recommended Usage

The recommended flow is:

```python
import sinr_cdf_utils as ncu
from SceneConfigSionna import SceneConfigSionna

results = ncu.run_two_mode_sinr_cdf_experiment(
    SceneConfig,
    num_macro_sims=N,
    ntn_drop_counts=[100, 200, 300],
    compute_positions_kwargs=compute_positions_kwargs,
    compute_cir_kwargs=compute_cir_kwargs,
    h_tn_th=h_tn_th,
    bs_tx_power=bs_tx_power,
    tn_tx_power=tn_tx_power,
    ntn_tx_power=ntn_tx_power,
    tn_noise_power=tn_noise_power,
    bs_noise_power=bs_noise_power,
)

metrics_path = ncu.save_two_mode_sinr_metrics(
    results,
    result_dir="result",
    output_name="two_mode_sinr_metrics.npz",
)
```

Plotting is now kept in `SINR_CDF.ipynb`, so you can edit titles, styles, and save paths directly in the notebook.

This produces:

- three SINR curves in each figure for `NTN = 100, 200, 300`
- saved metrics in `result/`

## Outputs

The two-mode helper writes:

- `result/two_mode_sinr_metrics.npz`
- `result/two_mode_sinr_cdf_mode1.png`
- `result/two_mode_sinr_cdf_mode1.pdf`
- `result/two_mode_sinr_cdf_mode2.png`
- `result/two_mode_sinr_cdf_mode2.pdf`

The older single-flow helper writes:

- `result/sinr_cdf_metrics.npz`

## Dependencies

The code expects a working local environment with at least:

- Python
- NumPy
- SciPy
- Matplotlib
- TensorFlow
- Sionna
- Mitsuba

It also assumes access to the scene assets referenced by the notebooks, for example the local XML scene used by `load_scene(...)`.

## Notes

- `SINR_CDF.ipynb` is the notebook name after the repository rename cleanup.
- `Pos_for_RM.ipynb` imports the renamed scene module as well.
- The repository currently keeps some older MUSIC/nulling helpers because they are still useful for comparison and debugging.
