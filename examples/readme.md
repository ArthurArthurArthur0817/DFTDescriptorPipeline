# Examples for Quantum Descriptor Regression 🧪

This folder contains **example data and walkthroughs** to demonstrate how to use the quantum descriptor extraction and regression pipeline from this repository.

It is designed to work directly with `extractor_regr.py` and the corresponding Jupyter notebooks (e.g., `azoarene.ipynb`, `indigo_aryl_alkyl.ipynb`, `indigo_diaryl.ipynb`, `heck_boronic_acids.ipynb`).

---

## 📁 Folder Contents

```

examples/
├── azoarene/
│   ├── azo\_data.xlsx             # Reaction rates and Ar1/Ar2 info
│   └── logs/                     # Gaussian .log files for each Ar group
│
├── indigo\_diaryl/
│   ├── diaryl\_data.xlsx
│   └── logs/
│
├── indigo\_aryl\_alkyl/
│   ├── indigo\_data.xlsx
│   └── logs/
│
├── heck\_boronic\_acids/
│   ├── heck\_data.xlsx
│   └── logs/

````

---

## 🧪 What You Can Do

Each subfolder is an independent example. You can:

1. Open the corresponding `.ipynb` notebook (e.g., `indigo_diaryl.ipynb`)
2. Point to the Excel file and `logs/` folder inside the example directory
3. Run the full pipeline using:

```python
from extractor_regr import run_full_pipeline

run_full_pipeline(
    log_folder='examples/indigo_diaryl/logs',
    xlsx_path='examples/indigo_diaryl/diaryl_data.xlsx',
    target='ln(kobs)',
    output_path='final_output.xlsx',
    plot_path='Regression_Plot.png'
)
````

---

## ✅ Example Output

For each example, the script will generate:

* `final_output.xlsx` – merged descriptors + targets
* `regression_search_results.csv` – all tested models
* `Regression_Plot.png` – visualization of best regression fit

---

## 📎 Notes

* Gaussian `.log` files must contain NBO, SCF, dipole, and polarizability sections.
* Ensure compound names in `.xlsx` match the corresponding `.log` filenames.
