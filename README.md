# DFTDescriptorPipeline

A fully automated pipeline for extracting quantum chemical descriptors (HOMO/LUMO, dipole moment, polarizability, Sterimol) from Gaussian log files and modeling structure–property or structure–reactivity relationships.

This project enables computational chemists to rapidly process `.log` files from DFT calculations and build interpretable linear models with minimal manual work.

---

## 🚀 Features

- 🧠 Auto-parses HOMO/LUMO energies, dipole moments, polarizability  
- 📦 Automatically identifies atoms for Sterimol descriptor computation  
- 📈 Compiles descriptors into CSV format ready for modeling  
- 💻 Jupyter-ready and modular  
- 🧪 Validated on Indigo/Azo photoswitches and Heck coupling case studies  

---

## 📂 Project Structure

```
DFTDescriptorPipeline/
├── descriptors/              # Core Python modules
│   ├── extractor.py          # HOMO/LUMO, dipole, polarizability extraction
│   └── sterimol.py           # Anchor atom identification for Sterimol
├── logfiles/                 # Example Gaussian log files
├── notebooks/                # Example analysis notebooks
├── requirements.txt          # Dependency list
└── README.md                 # This file
```

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/DFTDescriptorPipeline.git
cd DFTDescriptorPipeline
pip install -r requirements.txt
```

---

## 🧪 Quick Start

1. Place your `.log` files into the `logfiles/` folder.  
2. Open and run `notebooks/example_pipeline.ipynb`.

This will:

- Extract all descriptors  
- Compile them into a `descriptors.csv` file  
- Run a regression model (e.g., MLR) on the results  

---

## 🧩 Modules

You can use these functions directly in your own pipeline:

```python
from descriptors.extractor import extract_homo_lumo, extract_dipole_moment, extract_polarizability
from descriptors.sterimol import find_oh_bonds, find_c1_c2
```

---

## 📊 Example Applications

- Structure–reactivity modeling for boronic acids (Heck reaction)  
- Absorption tuning of indigo/azo photoswitches  
- Descriptor-based mechanistic hypothesis testing  

---

## 📄 License

MIT License

---

## 👩‍🔬 Citation

If you use this pipeline in your work, please cite the corresponding article in *Journal of Chemical Information and Modeling* (link will be added after acceptance).

## 🧪 Run on Google Colab

Click the badge below to run the example notebook on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/DFTDescriptorPipeline/blob/main/notebooks/example_pipeline.ipynb)