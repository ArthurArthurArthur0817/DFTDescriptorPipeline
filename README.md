# DFTDescriptorPipeline

A Python package for automated quantum chemical descriptor extraction from Gaussian log files and predictive modeling.

## 📦 Installation

```bash
pip install git+https://github.com/yourusername/DFTDescriptorPipeline.git
```

## 🚀 Quick Start

```python
from DFTDescriptorPipeline.aggregate import generate_feature_table
from DFTDescriptorPipeline.regression import run_regression

df = generate_feature_table("examples/heck_boronic_acids/logfiles", "examples/heck_boronic_acids/data.xlsx")
run_regression(df, target="ddG")
```

## 📁 Examples

Each folder in `examples/` contains:
- A set of Gaussian log files (`logfiles/`)
- A `data.xlsx` file with experimental data
- A `run.py` script to execute the pipeline

## 📘 Documentation

- `generate_feature_table(log_folder, excel_path)` - Extract descriptors
- `run_regression(df, target)` - Train and visualize regression model

## 🧪 Included Case Studies

- Heck boronic acids
- Azoarene photoswitches
- N-aryl-N’-alkyl indigo photoswitches
- N,N’-diaryl indigo photoswitches