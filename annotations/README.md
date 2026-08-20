# Annotation pipeline

This package generates independent Gemini + SAM 2 pseudo-annotations for RiskSight evaluation. The active reproducible study is:

```bash
python -m benchmarks.run_50_frame_study
```

Use the same command after interruption; compatible Gemini and SAM results are cached. Do not use `--force` unless paid regeneration is intentional. See `docs/AI_ANNOTATION.md` for the labeling policy, quality gates, evaluation method, and limitations.
