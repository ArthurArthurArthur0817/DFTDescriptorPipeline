# DFTDescriptorPipeline

**DFTDescriptorPipeline** is a Python toolkit for automated extraction of molecular descriptors from quantum chemistry log files and streamlined regression analysis. It is designed for computational chemistry and machine learning practitioners to efficiently convert raw DFT output logs into feature-rich datasets and perform downstream property prediction tasks.

## Features

- **Automated Descriptor Extraction:**  
  Parse Gaussian and other quantum chemistry log files to extract frequencies, NBO charges, and more.
- **Modular Regression Workflow:**  
  Build regression models (e.g., for property prediction) using extracted descriptors.
- **Batch Processing:**  
  Handle multiple files and molecules in a single workflow.
- **Flexible Output:**  
  Results are exported as clean CSV files for direct use in machine learning or statistical analysis.
- **Colab-Friendly:**  
  Run the pipeline end-to-end in Google Colab with minimal setup.

---

## Installation

The recommended way to use this package is via [Google Colab](https://colab.research.google.com/):

### Colab Usage

1. **Upload the Repository:**  
   Download and unzip the repository on your computer, then upload all files to your Colab environment.

2. **Install Required Packages:**  
   In a Colab cell, install required Python packages:
   ```python
   !pip install pandas numpy openpyxl
   ```

3. **Upload Input Files:**
   Upload your quantum chemistry log files (`*.log`) and Excel templates if needed.

---

### Local Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/DFTDescriptorPipeline.git
cd DFTDescriptorPipeline
pip install -r requirements.txt
```

---

## Usage

### Quickstart (Colab)

1. **Import and Run Extraction**

   ```python
   from extractor_regr import batch_extract_and_save
   batch_extract_and_save('/content/logs/', '/content/output/descriptors.csv')
   ```

2. **Perform Regression Analysis**

   ```python
   from regressor import run_regression_analysis
   run_regression_analysis('/content/output/descriptors.csv')
   ```

---

## Main Components & Functions

### `extractor_regr.py`

* **Purpose:**
  Extracts molecular descriptors from a batch of quantum chemistry log files and saves them as a CSV.
* **Main Functions:**

  * `batch_extract_and_save(input_dir, output_csv)`
    Process all log files in the specified directory and write a structured descriptor table.
  * `extract_frequencies(log_path)`
    Extracts vibrational frequencies from a single log file.
  * `extract_nbo_charges(log_path)`
    Extracts Natural Bond Orbital (NBO) charges from a single log file.
  * `extract_other_features(log_path)`
    Additional feature extraction (e.g., dipole moments, orbital energies).

### `regressor.py`

* **Purpose:**
  Builds and evaluates regression models (e.g., linear regression, random forest) using the extracted descriptors.
* **Main Functions:**

  * `run_regression_analysis(csv_path)`
    Loads the descriptor dataset, splits data, trains a regression model, and reports results (e.g., MAE, R²).
  * `feature_importance(model, X, y)`
    Displays ranked feature importances.
  * `plot_results(y_true, y_pred)`
    Plots prediction vs actual values for visual assessment.

### `examples/`

* **DFTDescriptorPipeline/examples/**
  Contains example input files and demo scripts for rapid onboarding.

---

## Example Workflow

#### 1. Upload log files to `/content/logs/` in Colab.

#### 2. Run the batch extraction:

```python
from extractor_regr import batch_extract_and_save
batch_extract_and_save('/content/logs/', '/content/descriptors.csv')
```

#### 3. Perform regression analysis:

```python
from regressor import run_regression_analysis
run_regression_analysis('/content/descriptors.csv')
```

---

## Dependencies

* Python 3.8+
* pandas
* numpy
* openpyxl
* (Optional: scikit-learn, matplotlib for regression/plotting)

Install all dependencies with:

```bash
pip install pandas numpy openpyxl scikit-learn matplotlib
```

---

## Customization & Extension

* **Add new descriptors:**
  Modify `extractor_regr.py` to add functions that extract additional features from log files.
* **Advanced regression models:**
  Edit `regressor.py` to implement custom models or additional evaluation metrics.
* **Pipeline integration:**
  Integrate with Jupyter/Colab notebooks or external ML frameworks as needed.

---

## License

[MIT License](LICENSE)

---