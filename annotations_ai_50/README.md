# 50-frame AI pseudo-annotation study

This directory contains the current drone evaluation dataset. Frames were sampled uniformly from `data/Drone.mp4`; Gemini performed semantic obstacle selection and local SAM 2 produced binary masks. These annotations were generated independently of RiskSight and are **not human-verified ground truth**.

`metadata.json` records generation configuration and cost, `review_needed.json` records excluded frames, and each frame JSON records whether it is usable for evaluation. The definitive held-out result is `outputs/benchmarks/obstacle_detection_drone_ai_50.json`.
