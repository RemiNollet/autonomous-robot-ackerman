# M2 results

config_hash=d5832461132b  (perception/model/training_config.yaml, epochs=60 lr=0.001 batch_size=64 seed=42)

No early stopping on val loss: v0's val split does not measure generalisation (ADR-11 finding 5 -- every arc is geometrically identical to its twin), so stopping on it would halt on noise and make runs non-reproducible. Epoch count is fixed, chosen from where perception/model/loss_curves.png actually plateaus.

kappa's loss weight is 0.0: the point-wise Frenet kappa label is measurably wrong within L_usable (2.36m) of a curvature transition (~42% of the loop). perception/model/kappa_transition_proximity.png: on straight samples, mean|kappa_pred| correlates at r=-0.67 with distance to the next transition, plateauing almost exactly at L_usable (0.023 beyond it vs 0.115 within it, a 5x gap) -- the network is reading the road correctly and being penalised for it. The MAE/scatter numbers below are NOT merely unusable: a kappa output that predicts curvature on straights and near-zero in curves would actively steer an MPC feedforward term off a straight line approaching a bend. The output head stays in the architecture (component_losses still logs kappa's raw, unweighted loss in loss_curves.png for monitoring) so a future windowed or continuous-curvature label can reuse it.

Plateau epoch per component (val, main model): {'e_y': 60, 'e_psi': 58, 'kappa': 51, 'confidence': 60}

| Run | Params | MACs | val loss | test/eval MAE e_y (m) | MAE e_psi (rad) | MAE kappa (1/m) | conf acc | train time |
|---|---|---|---|---|---|---|---|---|
| main (width=1.0) | 417,940 | 33,229,312 | 0.0145 | 0.0223 | 0.0652 | 0.5331 | 0.997 | 521.2s |
