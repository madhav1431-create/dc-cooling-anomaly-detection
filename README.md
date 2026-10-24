# Deep Learning-Based Anomaly Detection in Data Center Cooling Systems
 
An LSTM autoencoder that learns what normal sensor behaviour looks like on nodes of a real supercomputer
and flags 6-hour windows that look unusual. It is evaluated against recorded Nagios health alerts.
The project also contains a sensor-selection study (1 to 22 sensors), a supervised-LSTM comparison and a
Streamlit replay dashboard.
 
> **Status: Phase 1 (M100 dataset) is complete and documented here.**
> The next phase moves to a different dataset that has labeled cooling faults (see [Roadmap](#roadmap)).
> This README will be updated when that happens.
 
## Key findings
 
1. **A few fan and ambient sensors work better than all 22 sensors.** On held-out nodes, a 7-sensor model had
   about twice the precision and less than half the false alarms of the 22-sensor model, while catching the same
   number of alert episodes (table below).
2. **Fan sensors are the best single inputs.** Power sensors gave the most false alarms (about 430 to 720 alarm
   bursts, against 80 to 110 for fans), probably because power follows the workload.
3. **A supervised LSTM did not generalise.** Trained on labeled windows (one sensor, and all 22 sensors), its
   validation PR-AUC stayed near the random level and its ROC-AUC on held-out nodes was below 0.5.
4. **The labels are the bottleneck.** The labels are Nagios health alerts, not confirmed cooling faults. About half
   of the test alert episodes were never detected by any model.
5. **Accuracy is misleading here.** About 99% of windows are normal, so "always predict normal" already scores
   98.3% to 99.5%. Precision, recall, false alarms and episodes caught are the metrics that matter.
## Data
 
- **Source:** the public CINECA Marconi100 (M100) telemetry dataset. <!-- TODO: add dataset link / citation -->
- **Scope:** rack 0, nine nodes with few alerts: 7, 8, 9, 10, 11, 13, 14, 16, 17. Sampling interval: 15 minutes.
- **22 sensor columns:**
| Group | Columns |
|---|---|
| Ambient (1) | `ambient` |
| Fan speeds (8) | `fan0_0`, `fan0_1`, `fan1_0`, `fan1_1`, `fan2_0`, `fan2_1`, `fan3_0`, `fan3_1` |
| Temperatures (7) | `p0_vdd_temp`, `p1_vdd_temp`, `dimm_temp`, `p0_core_temp`, `p1_core_temp`, `gpu_core_temp`, `gpu_mem_temp` |
| Power (5) | `p0_power`, `p1_power`, `total_power`, `ps0_input_power`, `ps1_input_power` |
| Other (1) | `fan_disk_power` |
 
- **Labels:** Nagios health-alert states: 0 (OK), 2 (CRITICAL), 3 (UNKNOWN), NA. These meanings are inferred
  and not confirmed. A window counts as anomalous if any of its 24 steps is state 2. Windows containing state 3 or NA
  are dropped from evaluation.
- The raw data is large and is **not included** in this repository.
## Method
 
- **Windows:** 24 consecutive steps (6 hours), contiguous in time. Stride 4 for training and validation, stride 2 for test.
- **Split:** chronological 70% / 10% / 20% per node (train / validation / test).
  Window counts: train 104,640 (2,716 anomalous, 143 episodes), validation 18,214 (288, 35 episodes),
  test 72,252 (761, 31 episodes).
- **Preprocessing:** forward-fill up to 4 steps. The scaler is fit on **normal training rows only** and the result is
  clipped to ±5.
- **Model:** `LSTM(64) → Dense(16) → RepeatVector(24) → LSTM(64, return_sequences) → TimeDistributed(Dense(k))`.
  Adam (lr 1e-3), MSE loss, batch size 256, early stopping (patience 5, best weights restored), seed 42.
  Trained on normal windows only.
- **Anomaly score:** mean squared reconstruction error of a window.
- **Alarm threshold:** the 99th percentile of validation scores (95th also available). It is **never** taken from test data.
- **Metrics:** ROC-AUC, PR-AUC (random baseline = anomaly rate), precision, recall, F1, confusion counts, and
  episode-level results: alert episodes with at least one alarm inside, and "false alarm bursts" (runs of consecutive
  alarms with no recorded alert).
## Results
 
### 1. Original model (all 22 sensors, whole test set, 99th-percentile threshold)
 
| Score | ROC-AUC | PR-AUC | Episodes caught | Precision | False bursts |
|---|---|---|---|---|---|
| Full 22-sensor score | 0.787 | 0.0581 (random 0.0105) | 14/31 | 0.062 | 468 |
| Cooling-only score (16 sensors) | n/a | 0.0794 | 14/31 | 0.077 | 324 |
 
### 2. Sensor study
 
Each row is a **separate model trained only on those sensors**. A first attempt to select sensors on one half of the
test period failed, because almost all anomalies sit late in the period. The study instead selects on nodes
11, 13, 14, 16, 17 ("selection nodes", 16 episodes) and reports on nodes 7, 8, 9, 10 ("report nodes", 15 episodes).
Sensors are added best-first: `fan3_1`, `fan1_1`, `fan1_0`, `ambient`, `fan2_1`, `fan2_0`, `fan3_0`, `p1_vdd_temp`.
 
Report nodes (always-normal accuracy: 98.28%):
 
| Sensors | Precision | Recall | Episodes | False bursts | False positives (windows) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 1 (`fan3_1`) | 15.7% | 41.8% | 9/15 | 99 | 1,247 | 0.774 | 0.0853 |
| 7 (fans + ambient) | 18.3% | 43.6% | 9/15 | 91 | 1,087 | 0.725 | 0.1003 |
| 16 cooling sensors | 10.2% | 42.4% | 9/15 | 171 | 2,072 | 0.782 | 0.1421 |
| All 22 | 10.1% | 43.4% | 8/15 | 241 | 2,143 | 0.807 | 0.1033 |
 
More sensors did not catch more episodes and roughly doubled the false positives. Neighbouring rows differ by
amounts within training noise (a 3-sensor run collapsed to 2.2% precision; a repeat of a single-sensor model changed
its PR-AUC from 0.073 to 0.054 with the same seed, because GPU training is not perfectly repeatable).
 
**Final saved model (7 sensors, retrained and used by the dashboard):**
report nodes: ROC 0.724, PR 0.1004, precision 17.9%, recall 43.6%, 9/15 episodes, 93 false bursts;
selection nodes: ROC 0.667, PR 0.0705, precision 6.3%, recall 5.4%, 5/16 episodes, 33 false bursts.
 
### 3. Supervised LSTM vs autoencoder
 
A supervised LSTM classifier (anomaly weight about 6, Adam, up to 50 epochs, early stopping on validation PR-AUC)
was trained on labeled windows. Early stopping ended training at 20 and 21 epochs.
 
| Report nodes | Autoencoder | Supervised (`fan1_1`) | Autoencoder (22) | Supervised (22) |
|---|---|---|---|---|
| Precision | 16.0% | 1.8% | 10.1% | 6.6% |
| Recall | 42.4% | 17.4% | 43.4% | 6.5% |
| ROC-AUC | 0.793 | 0.384 | 0.807 | 0.322 |
| PR-AUC | 0.1267 | 0.0402 | 0.1033 | 0.0259 |
 
Validation PR-AUC stayed at 0.015 to 0.027 (random 0.0158) while training PR-AUC reached 0.56: the supervised model
memorised the training period. Alert frequency also drops over time (2.6% of windows in training, 1.6% in
validation, 1.05% in test), so the alerts may change or may not be tied to these sensors.
 
## Dashboard
 
`dashboard/app_v2.py` (Streamlit) replays the test period of each node. It is a **replay of recorded data, not
live monitoring**.
 
- **Replay tab:** NORMAL / POSSIBLE ANOMALY status, score chart with the threshold, recorded alerts, latest sensor
  readings and each sensor's share of the score.
- **Detector performance tab:** accuracy, precision, recall, F1, false positives and the other confusion counts as
  percentages, for one node, the report nodes, the selection nodes or all nodes. A threshold control (99th / 95th
  percentile) shows the precision/recall trade-off.
- **Sensor study tab:** results of the separate models trained on one sensor or on several sensors.
Replay metrics use every 15-minute window, so they differ slightly from the official evaluation numbers above.
`dashboard/app.py` is the earlier, more complex draft.
 
## Repository layout
 
```
dashboard/   Streamlit apps (app_v2.py is the current one)
models/      lstm_ae.keras (22-sensor model), final7.keras + final7_meta.json (final 7-sensor model,
             scaler values and thresholds)
notebooks/   cooling_analysis.ipynb (per-sensor error analysis, preprocessing, scoring)
outputs/     result tables (sensor study, single-sensor results, event-level results, thresholds)
lstm_threshold_results.csv, node10_error_plot.png   early results
```
 
Not in the repository: the raw M100 data and the processed parquet files (too large, listed in `.gitignore`).
 
## Reproducing
 
Environment used: Windows 11, Python 3.12, TensorFlow 2.21, pandas 3.0, Streamlit, Altair, scikit-learn.
The sensor study was run on a Colab T4 GPU. A requirements file is not included yet.
 
1. Obtain the M100 data for rack 0 and build the reduced table of the 9 nodes and 22 sensors.
2. Preprocess, window and split as described under *Method*.
3. Train the autoencoder (or the per-sensor models) and score the validation and test windows.
4. Run the dashboard: `python -m streamlit run dashboard/app_v2.py` (needs `data/dashboard_final7.parquet`,
   built from the final 7-sensor model).
## Limitations
 
- The labels are health alerts, **not confirmed cooling faults**. "Possible anomaly" only means "unusual compared
  with this node's normal behaviour".
- Small test sets (31 episodes in total, about 15 per node group) and **one training run per model**: small
  differences are within noise. The study is exploratory.
- Node 10 holds most of the anomaly windows in the report group, so window-based numbers there are dominated by it.
- The node split reduces, but does not remove, optimism: nodes share the same rack and period.
- Both the autoencoder and the supervised LSTM stayed weak on this dataset. The data, not epochs or tuning,
  appears to be the limit.
## Roadmap
 
- Move to a dataset with **labeled cooling faults** and evaluate unsupervised and supervised models on it
  with the same time-based protocol.
- Re-use the pipeline, sensor-study design and dashboard where possible.
- Update this README, the models and the results when the new dataset is in place.
