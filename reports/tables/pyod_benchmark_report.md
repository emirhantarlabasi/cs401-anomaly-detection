# PyOD Benchmark Report

## Protocol
- train_normal / val_mixed / test_mixed
- label mapping: normal=0, attack=1
- models: ['ECOD', 'COPOD', 'HBOS']
- feature_transform: none
- hbos_bins_grid: [10]
- hbos_ensemble_bins: []
- min_recall: 0.6
- max_fpr: 0.35

## Outputs
- `reports\tables\pyod_validation_sweep.csv`
- `reports\tables\pyod_model_comparison_test.csv`
- `reports\tables\model_comparison_with_pyod.csv`