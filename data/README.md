# Data (not redistributed)

1. **WiFiSpectralJam** (Herzalla, Singh, Lunardi, Andreoni, 2026; arXiv:2608.15728). Kaggle dataset `daniaherzalla/radio-frequency-jamming`
   (shown on Kaggle as "WiFiSpectralJam Dataset"). Used by `notebooks/uav_detector_in_the_loop_v3.ipynb`; add it as the notebook input on Kaggle.
   The per-channel detector scores and file metadata derived from it are stored in `results/kaggle_v3/` (`scores_*.npz`, `meta_*.csv`),
   so the closed-loop policy study can be rerun without downloading the raw data.
2. **UAV OFDM jamming dataset** (Li, Pawlak, Price, Al Shamaileh, Niyaz, Paheding, Devabhaktuni, IEEE Access 10, 16859–16870, 2022,
   doi:10.1109/ACCESS.2022.3150020). Clone into this folder:
   `git clone https://github.com/michaelevol/uavs_jamming_detection data/uavs_jamming_detection`
