# PyOD Stability Check Report

## Coverage
- sample_sizes: [100000, 250000, 400000]
- split_modes: ['random', 'time']
- threshold_percentiles: [50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 97.0, 99.0]
- fusion_weight_grid: [0.2, 0.4, 0.6, 0.8]

## Leakage Sanity
- rows with overlap > 0: 0
- all threshold selections: validation_only

## Outputs
- `reports\tables\pyod_stability_runs.csv`
- `reports\tables\pyod_stability_summary.csv`
- `reports\tables\pyod_leakage_sanity.csv`