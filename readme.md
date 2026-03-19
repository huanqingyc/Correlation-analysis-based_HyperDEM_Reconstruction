# CAHR

This repository is both:

- a callable code library for detector error model (DEM) reconstruction
- an implementation of the paper *Reconstruction of detector error model for quantum error correction*

## Setup

```bash
conda env create -f environment.yml
conda activate qec
```

## Main APIs

- `experiments.run_correlation_analysis`
- `experiments.run_decode_from_files`
- `inference_with_correlation_analysis.cal_multi_body_correlations`
- `decoding.create_dem_from_analysis`

## Reproduce Paper Data

The shell script `run_surface_color.sh` is used to reproduce the paper's surface-code
and color-code experiment data.

```bash
./run_surface_color.sh
```

Useful examples:

```bash
./run_surface_color.sh --surface --ca-only
./run_surface_color.sh --surface --decode-only
./run_surface_color.sh --color --ca-only
./run_surface_color.sh --color --decode-only
```

## Outputs

Results are written to:

```text
data/{code_task}_d{distance}r{rounds}/{shots_analysis}_{inference_eps}/
```

Typical files:

- `correlation.json`
- `correlation.txt`
- `ler.txt`