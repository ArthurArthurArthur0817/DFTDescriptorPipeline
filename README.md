# DFTDescriptorPipeline

A Python pipeline for automating descriptor extraction from Gaussian log files and performing regression modeling for reaction rate prediction.

## 🧠 Overview
This tool extracts quantum chemistry descriptors (e.g., NBO charges, HOMO/LUMO, dipole moment, vibrational frequency, Sterimol parameters) from Gaussian `.log` files and merges them with an Excel dataset. It then performs regression modeling with LOOCV (leave-one-out cross-validation) to identify the best feature combinations for predicting a target variable (e.g., `ln(kobs)`).

## 🔧 Features
- **Automatic morfeus installation** for Sterimol calculation
- **Descriptor extraction** from Gaussian logs:
  - HOMO / LUMO energies
  - Dipole moment
  - Polarizability
  - NBO charges & bond analysis
  - Vibration frequency and IR intensity of specific bonds
  - Sterimol descriptors (L, B1, B5)
- **Smart pairing and merging** of Ar1/Ar2 substituents
- **Data cleaning & NaN filtering**
- **LOOCV regression modeling** and model ranking
- **Regression plot** with performance metrics
- **Problem reporting** for molecules with incomplete features

## 📁 Directory Structure
```text
├── descriptors/extractor_regr.py                               # Main pipeline script
├── examples/azoarene/Azoarene.xlsx                             # Input Excel with Azoarene compound info
├── examples/azoarene/logfiles/                                 # Folder containing .log files of Azoarene
├── examples/heck_boronic_acids/Heck_boronic_acid.xlsx          # Input Excel with heck_boronic_acids compound info
├── examples/heck_boronic_acids/logfiles/                       # Folder containing .log files of heck_boronic_acids
├── examples/indigo_aryl_alkyl/N_aryl_N_alkyl_indigo            # Input Excel with N_aryl_N_alkyl_indigo compound info
├── examples/indigo_aryl_alkyl/logfiles/                        # Folder containing .log files of indigo_aryl_alkyl
├── examples/indigo_diaryl/NN_diaryl_indigo.xlsx                # Input Excel with NN_diaryl_indigo compound info
├── examples/indigo_diaryl/logfiles/                            # Folder containing .log files of NN_diaryl_indigo
````

## 🏁 Quick Start

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn morfeus-ml
```

> The script will auto-install `morfeus-ml` if missing.

### 2. Prepare your data

* Put your `.log` files in a folder (e.g., `logfiles/`)
* Your Excel file (e.g., `data.xlsx`) should include columns:

  * `Compound`, `Ar1`, `Ar2`, `ln(kobs)` or other target variable

### 3. Run the pipeline

```python
from extractor_regr import run_full_pipeline

run_full_pipeline(
    log_folder='logfiles',
    xlsx_path='data.xlsx',
    target='ln(kobs)',
    output_path='final_output.xlsx',
    plot_path='Regression_Plot.png',
    auto_pairing=True
)
```

## 📊 Output

* `final_output.xlsx`: Cleaned and merged features
* `regression_search_results.csv`: LOOCV results of all feature combinations
* `Regression_Plot.png`: Scatter plot of experimental vs predicted values
* `problem_index_report.xlsx`: List of molecules with incomplete feature extraction

## 📌 Notes

* Requires Gaussian `.log` files with **NBO**, **polarizability**, **dipole**, and **frequency** information.
* Atom indices are auto-parsed using NBO bonding rules.
* Sterimol parameters use filtered `.xyz` files with specific atoms removed.

## 🧪 Example Output (Best Model)

```
✅ Best model: ['Ar1_Ar_NBO_C2', 'Ar2_Ar_NBO_C2', 'Ar2_Ar_NBO_-O', 'Ar2_Ar_v_C=O', 'Ar2_Ar_Ster_L']
Q² = 0.860 | R² = 0.896
```

## 📜 License

MIT License
