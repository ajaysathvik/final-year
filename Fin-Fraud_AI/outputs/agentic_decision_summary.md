# Agentic Decision Summary

The agentic workflow uses four evidence channels:
- clean classification utility
- adversarial robustness utility
- corruption-shift stability
- calibration quality

Recommended deployment regime: **Original**
- Selected classifier: `Random Forest`
- Agentic score: `0.9456`
- Decision: `Deploy now`

Dataset-level ranking:
- Original: score `0.9456`, classifier `Random Forest`, decision `Deploy now`
- Adv-CTGAN: score `0.8700`, classifier `Random Forest`, decision `Use as robustness-aware backup`
- CTGAN: score `0.6276`, classifier `Neural Network (MLP)`, decision `Keep as baseline only`
