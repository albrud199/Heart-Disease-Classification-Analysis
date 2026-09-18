# Phase 7 — Production Handoff Report

## Status
Complete: Steps 24–27 passed.

## Production contract
- Artifact: `models\phase5\hybrid_ensemble_calibrated_production.pkl`
- Locked threshold: `0.51`
- Calibration: `isotonic`
- SHA-256: `9b65cddd37ba0e321cf3783f22c433de25fc569f008ee5e5a1b512ea30d08a13`

## Locked Phase 5 test metrics
- roc_auc: 1.0
- roc_auc_ci95_low: 1.0
- roc_auc_ci95_high: 1.0
- pr_auc: 1.0
- pr_auc_ci95_low: 1.0
- pr_auc_ci95_high: 1.0
- accuracy: 1.0
- recall_sensitivity: 1.0
- specificity: 1.0
- precision: 1.0
- f1: 1.0
- balanced_accuracy: 1.0
- brier_calibrated: 0.0048780487814682
- ece_calibrated: 0.0097570780487803
- log_loss_calibrated: 0.0135258035236111
- brier_uncalibrated_reference: 0.0066962955674636
- ece_uncalibrated_reference: 0.0255852705397763

## Validation
- Input schema, probability range, threshold behavior, serialization round-trip, and row counts passed.
- No fitting, calibration, tuning, or threshold selection was performed in Phase 7.

## Limitations
- Small single-cohort dataset; no external or temporal validation.
- Test performance is an evaluation result, not a clinical performance guarantee.
- Research and educational use only; not a diagnostic device.
