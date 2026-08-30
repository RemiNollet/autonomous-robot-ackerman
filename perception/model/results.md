# M2 results

seed=42 epochs=40 lr=0.001 batch_size=64

| Run | Params | MACs | val loss | test/eval MAE e_y (m) | MAE e_psi (rad) | MAE kappa (1/m) | conf acc | train time |
|---|---|---|---|---|---|---|---|---|
| main (width=1.0) | 417,940 | 33,229,312 | 0.0513 | 0.0177 | 0.0587 | 0.2466 | 1.000 | 333.7s |
| constant predictor | 3 | 0 | n/a | 0.1745 | 0.2598 | 0.1385 | 0.865 | 0.0s |
| width ablation (width=0.25) | 65,512 | 2,835,712 | 0.0550 | 0.0350 | 0.0720 | 0.2347 | 1.000 | 276.4s |
| mirror-generalization probe | 417,940 | 33,229,312 | 0.0744 | 0.0453 | 0.1112 | 0.1627 | 0.999 | 198.3s |

*mirror-generalization probe*: trained on train-split SOURCE renders only; 'test' column here is evaluation on those same samples' MIRROR twins (unseen), not the usual test split
