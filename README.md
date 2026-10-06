# SPARK

Code accompanying the manuscript **“A Deep Learning Approach to Quantitative PCR that Learns from Ground Truth.”**

SPARK is a deep-learning model for classifying quantitative PCR (qPCR) amplification curves. This repository contains the model architecture, preprocessing, training, and inference code used for SPARK.

## Installation

```bash
python -m pip install -e .
```

## Inference

Input curves should be provided as a CSV containing `curve_id`, `gene`, and `fn_1` through `fn_40`.

```bash
spark-predict \
  --checkpoint /path/to/checkpoint.pth \
  --input /path/to/curves.csv \
  --output /path/to/predictions.csv
```

The output contains the predicted probability of amplification for each curve.

## Training

Training data should be provided as a CSV containing:

- `curve_id`
- `gene`
- `split` (`train` or `val`)
- `groundtruth_target`
- `expert_call`
- `fn_1` through at least `fn_40`

Train SPARK with:

```bash
spark-train \
  --input /path/to/training_curves.csv \
  --output-dir runs/my_assay
```

The training command uses the SPARK architecture and default training configuration described in the manuscript. Model checkpoints and normalization parameters are saved to the output directory.

See:

```bash
spark-train --help
```

for additional training options.
