# Detector-in-the-Loop Anti-Jamming for UAV Links

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24410565.svg)](https://doi.org/10.5281/zenodo.24410565)

Code, Kaggle notebook and results for a closed-loop study of UAV anti-jamming in which the jammer state is
**inferred by a detector trained on real spectral-scan measurements** instead of being given to the decision
policy as an oracle, as most anti-jamming studies assume.

Author: Renat Abdirayimov (ORCID [0009-0005-3636-8864](https://orcid.org/0009-0005-3636-8864)),
Tashkent University of Information Technologies named after Muhammad al-Khwarizmi.

## What is in this repository

| Path | Content |
|---|---|
| `notebooks/uav_detector_in_the_loop_v3.ipynb` | Kaggle notebook: per-channel jamming detector on WiFiSpectralJam passive scans (5 GHz and 2.4 GHz), trace-driven closed-loop emulator, baselines incl. DQN, saves detector-score artefacts |
| `src/emulator_and_policies.py` | Emulator and baseline policies extracted from the notebook (runs offline on the saved scores) |
| `src/feedback_policies.py` | NACK-hop and the proposed **Adaptive Feedback Threshold (AFT)** policy |
| `scripts/run_policy_study.py` | Reproduces `results/policy_study/` (closed-loop study, paired Wilcoxon tests, AFT sensitivity) |
| `scripts/li_uav_jamming_audit.py` | Leakage-aware audit of the public UAV jamming dataset of Li et al. (2022) |
| `results/kaggle_v1`, `results/kaggle_v3` | Raw outputs of the two Kaggle runs (detector metrics, closed-loop runs, scores, metadata) |
| `results/policy_study` | Offline policy study on the real detector scores |
| `results/li_audit` | Audit results on the Li et al. UAV dataset |
| `data/README.md` | Where to obtain the datasets (not redistributed) |

## Setup in brief

- **Sensing data.** WiFiSpectralJam (Raspberry Pi CM4 + QCA9880 spectral scans, HackRF jammer). Passive scans; per channel, 6 statistics of 7 driver fields plus channel-minus-median features; channel-agnostic HistGB detector.
- **Training domain.** Chamber floor + background locations 2–3 (benign), jammer powers +6/+9/+12 dBm.
- **Shifted domains.**
  - Unseen background location 1.
  - Unseen weak jammer (+3 dBm).
  - Both.
- **Closed loop (trace-driven emulation).**
  - One slot = one real scan file. When the jammer is ON, a real jammed file of the power class set by the UAV–jammer distance is used, with its target channel rotated onto the jammer's logical channel. When OFF, a real benign file is used.
  - Link: A2G free-space path loss, Rician K = 7.6 dB, SINR threshold 5 dB, spectral-overlap matrix (2.4 GHz overlapping, 5 GHz orthogonal channels).
  - Jammers: sweep, Markov, reactive.
- **AFT (proposed).**
  - The detector is frozen. Its decision threshold on log-score moves down after a NACK on a channel the detector declared clean, and relaxes back on ACKs.
  - A floor at the 99.9 % quantile of training benign scores bounds false alarms.
  - The UAV leaves a channel right after a NACK.

## Main results (PSR = packet success rate; 10 seeds × 3 jammers, T = 2000 slots)

| Policy | 5 GHz no shift | 5 GHz unseen +3 dBm | 2.4 GHz no shift | 2.4 GHz unseen +3 dBm |
|---|---|---|---|---|
| Oracle jammer state | 0.9994 | 0.9994 | 1.0000 | 1.0000 |
| Detector, 0.5 threshold | 0.9992 | 0.9356 | 0.9997 | 0.9882 |
| FBR (online Platt) | 0.9992 | 0.9599 | 0.9997 | 0.9883 |
| FCCS (conformal) | 0.9738 | 0.9723 | 0.9667 | 0.9660 |
| Detector + NACK-hop | 0.9992 | 0.9711 | 0.9997 | 0.9906 |
| AFT without NACK-hop | 0.9992 | 0.9736 | 0.9997 | 0.9886 |
| **AFT (proposed)** | **0.9992** | **0.9815** | **0.9997** | **0.9910** |

- **Detector under the weak-jammer shift (5 GHz).** AUROC stays 0.945, but true-positive rate at 0.5 falls to 0.9 %. The failure is threshold transfer, not lost ranking.
- **Oracle-state overstatement (5 GHz, +3 dBm).** 6.4 pp on average and 17 pp for the reactive jammer (0.829 vs 0.999). AFT reduces the average gap to 1.8 pp, with no loss when there is no shift.
- **Statistics (5 GHz, +3 dBm).** AFT vs detector: +4.6 pp, 28 wins, 1 tie, 1 loss of 30 paired runs, Wilcoxon p = 4·10⁻⁶. Across 27 parameter settings, AFT PSR is 0.980–0.984.
- **Unseen background location.** No measurable effect on the detector or on any policy.
- **Li et al. UAV dataset audit (RF, 4-class jamming type).**

  | Split | Accuracy |
  |---|---|
  | Random 70/30 | 0.899 |
  | Temporal blocks | 0.691 |
  | Unseen jammer location | 0.855 |
  | Unseen separation | 0.768 |

  Unseen jamming types are always flagged as jamming but absorbed into known types.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_policy_study.py            # ~6 min CPU, uses results/kaggle_v3 artefacts
git clone https://github.com/michaelevol/uavs_jamming_detection data/uavs_jamming_detection
python scripts/li_uav_jamming_audit.py        # ~2 min CPU
```

To regenerate the detector scores from raw data, run `notebooks/uav_detector_in_the_loop_v3.ipynb` on Kaggle with the
dataset `daniaherzalla/radio-frequency-jamming` attached (CPU only; several hours on the free Kaggle CPU).

## Limitations

- Closed-loop results come from trace-driven emulation with real sensing data, not from flight measurements.
- Jamming captures are chamber recordings at 20 cm.
- The target channel is rotated across logical channels.
- Power classes are assigned by distance quantiles.

## Citation

See `CITATION.cff`. Archived on Zenodo: [10.5281/zenodo.24410565](https://doi.org/10.5281/zenodo.24410565) (v1.0.0).
Please also cite the datasets: Herzalla et al., WiFiSpectralJam, arXiv:2608.15728 (2026); Li et al., IEEE Access 10, 16859–16870 (2022),
doi:10.1109/ACCESS.2022.3150020.

## License

MIT (code). Results files are released under CC BY 4.0.
