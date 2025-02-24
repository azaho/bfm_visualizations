# BFM-vis

## Getting Started

Optionally, create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

1. Install required packages:
```
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil librosa
```

2. Download and extract the braintreebank dataset (this step can be skipped if the dataset is already downloaded and extracted; it should be all extracted into the braintreebank/ directory):
```
python braintreebank_download_extract.py
```
alternatively, you can specify the path to the braintreebank dataset in the `btbench_config.py` file:
```
ROOT_DIR = "braintreebank" # Root directory for the braintreebank data
```
