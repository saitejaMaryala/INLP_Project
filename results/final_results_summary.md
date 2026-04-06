# Final Results Summary

This file consolidates the latest local runs before GitHub push.

## 1) Bias-in-Bios (from results/bios/bios_results.json)

- FP32:
  - accuracy: 0.73648
  - macro_f1: 0.66900
  - mean_abs_eod: 0.13504
  - max_abs_eod: 0.52013
- FP16:
  - accuracy: 0.73648
  - macro_f1: 0.66889
  - mean_abs_eod: 0.13483
  - max_abs_eod: 0.52077
- INT8:
  - accuracy: 0.59955
  - macro_f1: 0.54424
  - mean_abs_eod: 0.15544
  - max_abs_eod: 0.40743

Calibration mode metadata is stored in `results/bios/bios_results.json` under `quantization_meta`.

## 2) Jigsaw (from latest local run logs)

- FP32:
  - accuracy: 0.8945
  - macro_f1: 0.6685
  - mean_DPD: 0.1418
  - mean_EOD: 0.2455
- FP16:
  - accuracy: 0.8947
  - macro_f1: 0.6689
  - mean_DPD: 0.1418
  - mean_EOD: 0.2455
- INT8:
  - accuracy: 0.9262
  - macro_f1: 0.5571
  - mean_DPD: 0.0129
  - mean_EOD: 0.0753

Quantization mode status from latest run:

- requested_mode: static
- applied_mode: dynamic (fallback)
- fallback_reason: ONNX/export static PTQ path failed in current environment

## 3) Interpretation snapshot

- FP16 stays very close to FP32 on both tasks.
- INT8 changes the fairness/performance tradeoff strongly; in Jigsaw, lower DPD/EOD came with much lower macro-F1.
- Treat fairness improvements under INT8 carefully when utility drops.
