# Final Experiment Summary

## Frozen Configuration
- data_path: `data\cicids2017_cleaned.csv`
- row_limit: `250000`
- contamination: `0.36`
- ae_epochs: `20`
- seed: `42`
- ae_fixed_threshold: `0.005240097370565431`
- hybrid_fixed_threshold: `0.050350018691288925`
- default_hybrid_weight_if: `0.2`
- weight_grid: `[0.2, 0.4, 0.6, 0.8]`

## Produced Artifacts
- benchmark_all: `reports\tables\model_benchmark_all.csv`
- hybrid_weight_sweep: `reports\tables\hybrid_weight_sweep.csv`

## Official Hybrid Weight
- weight_if: `0.2`
- attack_f1: `0.017265677218808635`
- attack_recall: `0.011731103839966733`
- false_positive_rate: `0.20000126088299636`