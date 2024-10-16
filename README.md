# SAURL-TS: A Self-Adaptive Framework for Unsupervised Time Series Representation Learning

This repository contains the code for **SAURL-TS**, a framework for unsupervised time series representation learning. 

## Requirements

We use Python 3.9. The main packages include:

- `numpy`
- `scikit-learn`
- `torch`
- `matplotlib`
- `pandas`

You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```bash
python train.py
```

## Project Structure

```

├── datasets/                 # Directory for datasets
├── models/                   # Directory for models
├── tasks/                    # Directory for downstream task
├── config.py                 # Configuration file for tasks
├── data_load.py              # Script for loading datasets
├── requirements.txt          # Python packages
├── saurl_ts.py               # Main SAURL-TS model file
├── train.py                  # Main training script
├── utils.py                  # Utility functions
└── README.md                 # Documentation

```

---

