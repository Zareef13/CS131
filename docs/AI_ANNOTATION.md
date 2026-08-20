# AI-assisted obstacle annotations

RiskSight's current evaluation uses a two-stage annotation pipeline that is completely independent of RiskSight output:

```text
source frame → Gemini semantic boxes/points → local SAM 2 masks → quality gates
```

Gemini decides whether a frame is a usable forward-facing navigation view and identifies at most four solid obstacles in the current or near-future flight corridor. SAM 2 converts each box and positive point into a binary object mask. Menus, passenger views, static scenery, oversized merged masks, invalid masks, and other review cases are excluded from evaluation.

Every frame record preserves the model names, prompt, timestamp, token-derived cost, semantic coordinates, SAM candidate scores, selected candidate, rejection reasons, and evaluation eligibility. Gemini and SAM caches make interrupted runs restart-safe. RiskSight heatmaps are never supplied to either annotator.

## Reproduce the 50-frame study

Set `GEMINI_API_KEY`, install the `annotation` dependency group, and run:

```bash
python -m benchmarks.run_50_frame_study
```

The command samples 50 drone frames into `annotations_ai_50/`, resumes compatible cached work, applies quality gates, requires at least ten usable frames, selects a RiskSight threshold on alternating validation samples, and evaluates the frozen threshold on the remaining held-out samples.

To rerun only the local evaluation without API calls:

```bash
python -m benchmarks.evaluate_obstacle_detection \
  data/Drone.mp4 annotations_ai_50/drone \
  --width 640 --tolerance 10 \
  --output-name obstacle_detection_drone_ai_50
```

## Interpretation

The evaluator measures whether thresholded RiskSight responses cover each independently annotated obstacle, allowing a documented 10-pixel spatial tolerance. It also measures tolerant boundary localization. This is detector-oriented evaluation of a continuous risk heatmap, not semantic-segmentation evaluation.

All results are against **AI-generated pseudo-ground truth that has not been human-verified**. They demonstrate preliminary obstacle sensitivity only; they are not safety validation, calibrated collision probabilities, or evidence of generalization to other cameras or environments.
