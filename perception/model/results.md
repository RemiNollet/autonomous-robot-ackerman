# M2 results

config_hash=d5832461132b  (perception/model/training_config.yaml, epochs=60 lr=0.001 batch_size=64 seed=42)

No early stopping on val loss: v0's val split does not measure generalisation (ADR-11 finding 5 -- every arc is geometrically identical to its twin), so stopping on it would halt on noise and make runs non-reproducible. Epoch count is fixed, chosen from where perception/model/loss_curves.png actually plateaus.

kappa's loss weight is 0.0: the point-wise Frenet kappa label is measurably wrong within L_usable (2.36m) of a curvature transition (~42% of the loop). perception/model/kappa_transition_proximity.png: on straight samples, mean|kappa_pred| correlates at r=-0.67 with distance to the next transition, plateauing almost exactly at L_usable (0.023 beyond it vs 0.115 within it, a 5x gap) -- the network is reading the road correctly and being penalised for it. The MAE/scatter numbers below are NOT merely unusable: a kappa output that predicts curvature on straights and near-zero in curves would actively steer an MPC feedforward term off a straight line approaching a bend. The output head stays in the architecture (component_losses still logs kappa's raw, unweighted loss in loss_curves.png for monitoring) so a future windowed or continuous-curvature label can reuse it.

Plateau epoch per component (val, main model): {'e_y': 60, 'e_psi': 59, 'kappa': 57, 'confidence': 60}

| Run | Params | MACs | val loss | normalized test MAE e_y | normalized test MAE e_psi | conf acc | train time |
|---|---|---|---|---|---|---|---|
| main (width=1.0) | 417,940 | 33,229,312 | 0.0147 | 0.0219 | 0.0595 | 0.995 | 601.3s |
| constant predictor | 3 | 0 | n/a | 0.1833 | 0.2332 | 0.906 | 0.0s |
| mirror-generalization probe | 417,940 | 33,229,312 | 0.0177 | 0.0440 | 0.1133 | 0.998 | 275.8s |

*mirror-generalization probe*: trained on train-split SOURCE renders only; 'test' column here is evaluation on those same samples' MIRROR twins (unseen), not the usual test split

## Test-set error in physical units

See `perception/model/physical_metrics.py`. "Normalized" above is the same unit the loss is computed in (dimensionless, scaled by envelope width); the tables below are the physical-unit numbers a controller would actually see.

### main (width=1.0), test split
**In-distribution error, not a generalisation measurement.** Per ADR-11 finding 5, no partition of v0 measures generalisation -- every arc is geometrically identical to its twins under identical lighting and texture. These are upper bounds on a track the model has effectively memorised, reported because they're still the honest description of what was measured, not because they say whether the model can drive on a track it hasn't seen.
| Output | Units | MAE | RMSE | p50 | p95 | max | n |
|---|---|---|---|---|---|---|---|
| e_y | m | 0.0219 | 0.0286 | 0.0179 | 0.0581 | 0.1058 | 348 |
| e_psi | rad | 0.0595 | 0.0722 | 0.0531 | 0.1358 | 0.1865 | 348 |
| e_psi | deg | 3.41 | 4.14 | 3.04 | 7.78 | 10.69 | 348 |

e_y MAE is 5.5% of the lane half-width (0.4 m); p95 is 14.5%. e_psi MAE is 11.9% of the sampling envelope (+/-0.5 rad = +/-28.6 deg); p95 is 27.2%.

Confidence: accuracy 0.995 over n_valid=348, n_invalid=36 (imbalanced ~90/10 -- accuracy alone hides class performance). Per-class: valid recall 1.000, invalid recall 0.944.

**kappa: not reported.** Loss weight is 0 (ADR-12) -- the head is untrained, so any number here would describe initialization drift, not model performance.

| Curvature bin | e_y MAE (m) | e_y p95 (m) | e_psi MAE (deg) | e_psi p95 (deg) | n |
|---|---|---|---|---|---|
| straight | 0.0191 | 0.0473 | 2.92 | 6.25 | 138 |
| R=5m arc | 0.0232 | 0.0557 | 3.54 | 7.54 | 126 |
| R=3m arc | 0.0247 | 0.0732 | 4.02 | 8.61 | 84 |

### mirror-generalization probe, mirror-twin eval set
**The one v0 number that is not pure interpolation.** Trained on source (non-mirrored) renders only, evaluated on their never-seen mirror twins (right turns from a track that only physically contains left turns, ADR-10) -- geometrically distinct enough from training that this measures something closer to generalisation than the in-distribution table above does.
| Output | Units | MAE | RMSE | p50 | p95 | max | n |
|---|---|---|---|---|---|---|---|
| e_y | m | 0.0440 | 0.0583 | 0.0342 | 0.1144 | 0.4211 | 1273 |
| e_psi | rad | 0.1133 | 0.1460 | 0.0894 | 0.2836 | 0.3753 | 1273 |
| e_psi | deg | 6.49 | 8.36 | 5.12 | 16.25 | 21.50 | 1273 |

e_y MAE is 11.0% of the lane half-width (0.4 m); p95 is 28.6%. e_psi MAE is 22.7% of the sampling envelope (+/-0.5 rad = +/-28.6 deg); p95 is 56.7%.

Confidence: accuracy 0.998 over n_valid=1273, n_invalid=144 (imbalanced ~90/10 -- accuracy alone hides class performance). Per-class: valid recall 1.000, invalid recall 0.979.

**kappa: not reported.** Loss weight is 0 (ADR-12) -- the head is untrained, so any number here would describe initialization drift, not model performance.

| Curvature bin | e_y MAE (m) | e_y p95 (m) | e_psi MAE (deg) | e_psi p95 (deg) | n |
|---|---|---|---|---|---|
| straight | 0.0266 | 0.0593 | 2.21 | 5.47 | 601 |
| R=5m arc | 0.0499 | 0.1121 | 8.88 | 14.84 | 416 |
| R=3m arc | 0.0751 | 0.1379 | 12.66 | 19.44 | 256 |
