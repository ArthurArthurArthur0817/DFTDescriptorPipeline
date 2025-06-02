# DFTDescriptorPipeline Colab Case Studies

This repository includes four real-world case studies that illustrate how to use DFTDescriptorPipeline for quantum chemistry descriptor extraction and regression analysis.
**Each case can be run directly in Google Colab.** Click any folder link below to open the corresponding example.

---

## Case Study Overview

* No code modifications needed—simply switch datasets or problem types.
* All workflows start from quantum chemistry log files and end with ready-to-use regression models.
* Each folder below contains a complete Colab notebook and required input data.

---

### [1. Indigo Photoswitches: N,N'-Diaryl](indigo_diaryl/)

* Predict Z→E isomerization barriers for symmetric N,N'-diaryl indigo derivatives.
* Uses electronic descriptors to achieve strong model performance.
* Output includes feature tables and regression plots.

---

### [2. Indigo Photoswitches: N-Aryl-N'-Alkyl](indigo_aryl_alkyl/)

* Model isomerization barriers for unsymmetrical N-aryl-N'-alkyl indigo switches.
* Combines steric (Sterimol L) and electronic descriptors for improved accuracy.
* Output includes model metrics and visualizations.

---

### [3. Azoarene Photoswitches](azoarene/)

* Predict thermal isomerization rates of substituted azoarenes.
* Employs both geometric (bond lengths) and electronic features.
* Notebook outputs feature importance and predictive performance.

---

### [4. Heck Boronic Acids](heck_boronic_acids/)

* Predict reaction yields for para- and meta-substituted boronic acids in Heck coupling.
* Features include dipole moment, polarizability, LUMO energy, and IR intensity.
* Results include regression models and summary plots.

---