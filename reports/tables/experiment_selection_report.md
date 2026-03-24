# Experiment Selection Report

## Protocol
- train_normal / val_mixed / test_mixed split
- label mapping: normal=0, attack=1
- min_recall rule: 0.6

## Best Validation Candidates
- IF: {'contamination': 0.3, 'max_samples': 512.0, 'n_estimators': 100.0, 'accuracy': 0.5101015320226315, 'attack_precision': 0.6433238583927056, 'attack_recall': 0.3814182534471438, 'attack_f1': 0.47890188103711234, 'normal_recall': 0.6954356323288362, 'macro_f1': 0.5083390439689334, 'false_positive_rate': 0.3045643676711638, 'training_time_s': 0.6268241999932798, 'inference_time_s': 0.266133400000399}
- AE: {'batch_size': 512.0, 'epochs': 20.0, 'hidden_dim': 64.0, 'latent_dim': 16.0, 'lr': 0.001, 'threshold_percentile': 90.0, 'threshold_value': 0.03540394070212337, 'accuracy': 0.946392125455344, 'attack_precision': 0.9324200532978014, 'attack_recall': 0.9802144889472533, 'attack_f1': 0.9557201084057105, 'normal_recall': 0.8976799899129996, 'macro_f1': 0.9439026686760658, 'false_positive_rate': 0.10232001008700038, 'training_time_s': 16.089970600005472, 'inference_time_s': 0.0174228999967454}
- Hybrid: {'weight_if': 0.2, 'weight_ae': 0.8, 'threshold_percentile': 60.0, 'threshold_value': 0.015339331553263325, 'accuracy': 0.8138192058284032, 'attack_precision': 0.7742297508197871, 'attack_recall': 0.9663383672576056, 'attack_f1': 0.8596824284198331, 'normal_recall': 0.5941558441558441, 'macro_f1': 0.7915498745695335, 'false_positive_rate': 0.40584415584415584}

## Test Comparison File
- `reports\tables\experiment_model_comparison_test.csv`