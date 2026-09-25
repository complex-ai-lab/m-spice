# [KDD 2026] M-SPICE: Multimodal SPatIal Context for Epidemic Forecasting

Official implementation of ["Beyond Time Series: Spatial Reasoning for Epidemic Forecasting via Multimodal Learning"](https://dl.acm.org/doi/abs/10.1145/3770855.3819015) appearing in KDD 2026.

## Overview

M-SPICE is a multimodal epidemic forecasting framework that integrates temporal surveillance data with higher-resolution spatial context for multi-horizon epidemic forecasting.

This repository contains the code used to train and evaluate M-SPICE and the supported baseline models.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

The processed datasets and pretrained Stage 0 weights used in this work are available through Google Drive:

**Download:** [M-SPICE-data](https://drive.google.com/drive/folders/1c65rssl1AVgM1ZIQtBj9Vun03PnoFs_b?usp=drive_link)

After downloading, place the processed datasets in `data/` directory and the pretrained weights in `stage0/`

## Repository Structure

```text
.
├── data/                  # Processed datasets
├── results/               # Model outputs and evaluation results
├── setup/                 # Dataset, model, and experiment configurations
├── src/
│   └── forecaster/        # Training and evaluation code
├── stage0/                # Pretrained weights from Stage 0 of M-SPICE
├── requirements.txt
└── README.md
```

## Usage

Run the following commands from the `forecaster/` directory.

### Training

To train M-SPICE using experiment configuration `3`:

```bash
PYTHONPATH=.. python online_training.py --input=3
```

Experiment configurations are defined in:

```text
setup/exp_params/
```

The `--input` argument specifies the experiment configuration to use.

### Evaluation

To evaluate a trained model:

```bash
PYTHONPATH=.. python eval.py --input=3
```

Use the same experiment configuration used during training.

## Configuration

Experiment settings are specified using YAML configuration files.

```text
setup/
├── exp_params/            # Experiment-specific configurations
└── covid_mortality_3.yaml # Data and model configurations
```

Dataset, model, and training parameters can be modified through these configuration files.


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{gomez2026beyond,
author = {Gomez, Diana Guadalupe and Wu, Chenwei and Wang, Zhiyi and Shen, Liyue and Rodr{\'i}guez, Alexander},
title = {Beyond Time Series: Spatial Reasoning for Epidemic Forecasting via Multimodal Learning},
year = {2026},
isbn = {9798400722592},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3770855.3819015},
numpages = {12},
keywords = {multimodal learning, time series forecasting, spatiotemporal machine learning, epidemiology, public health},
location = {Republic of Korea},
series = {KDD '26}
}
```

## License

<LICENSE INFORMATION>
