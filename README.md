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

## Example: public qPCR data

A small demo using the public `C60.amp` dataset from the [`chipPCR` package](https://cran.r-project.org/package=chipPCR) is included to test the preprocessing and inference workflow end to end.

The demo requires `Rscript` and a SPARK checkpoint available locally:

```bash
python examples/run_c60_amp.py --checkpoint /path/to/checkpoint.pth
```

Results are written to `runs/c60_amp/`.

This example is intended as a functional demonstration of the code, not as a full reproduction of the paper's external-validation analysis. It follows the preprocessing used for these external data, including normalization from the unlabeled example curves and use of the E-gene model input for both assays. The latter is only a model encoding and does not imply that Vimentin or MLC-2v is an E gene.
