# DFTDescriptorPipeline Colab Tutorial

Welcome!  
This user manual will guide you through four practical examples using DFTDescriptorPipeline, each demonstrated as a Google Colab notebook.  
**By following these step-by-step notebooks, you will quickly master descriptor extraction, batch processing, regression analysis, and pipeline customization for quantum chemistry data.**

---

## Table of Contents

- [Before You Start](#before-you-start)
- [Example 1: Single Log File Descriptor Extraction](#example-1-single-log-file-descriptor-extraction)
- [Example 2: Batch Log File Processing](#example-2-batch-log-file-processing)
- [Example 3: Regression Analysis with Extracted Descriptors](#example-3-regression-analysis-with-extracted-descriptors)
- [Example 4: Extending the Pipeline with Custom Features](#example-4-extending-the-pipeline-with-custom-features)
- [FAQ and Troubleshooting](#faq-and-troubleshooting)
- [Contact](#contact)

---

## Before You Start

- **Recommended:**  
  All examples are provided as `.ipynb` notebooks under the `examples/` directory.
- **Run in Google Colab:**  
  Simply open each notebook in [Google Colab](https://colab.research.google.com/).  
  > You can do this by right-clicking on the notebook file in your GitHub repo and selecting "Open in Colab" (if enabled), or uploading the notebook to Colab directly.

- **Dependencies:**  
  Each notebook will auto-install necessary Python packages (e.g., `pandas`, `numpy`, `openpyxl`).  
  If not, add the following cell at the top:
  ```python
  !pip install pandas numpy openpyxl scikit-learn matplotlib
  ```

---

## Example 1: Single Log File Descriptor Extraction

**Path:** `examples/example1_single_log/extract_single_log.ipynb`

* **Objective:**
  Learn how to extract descriptors from a single quantum chemistry log file and export them as a CSV.
* **What You Will Do:**

  * Upload a sample `.log` file.
  * Run extraction cells to generate a CSV of descriptors.
  * Inspect the output.

**How to Use:**

1. Open `extract_single_log.ipynb` in Colab.
2. Upload the sample log file when prompted.
3. Execute all cells in order.
4. Download or view the resulting CSV file with molecular descriptors.

---

## Example 2: Batch Log File Processing

**Path:** `examples/example2_batch_logs/batch_process_logs.ipynb`

* **Objective:**
  Batch process a folder of log files to automatically extract descriptors for multiple molecules.
* **What You Will Do:**

  * Upload a directory of log files.
  * Run the batch extraction notebook.
  * Combine results into a single CSV file.

**How to Use:**

1. Open `batch_process_logs.ipynb` in Colab.
2. Upload your set of log files (you may zip them for easier upload).
3. Execute the notebook cells to extract and merge descriptors.
4. Inspect or download the compiled CSV.

---

## Example 3: Regression Analysis with Extracted Descriptors

**Path:** `examples/example3_regression/regression_analysis.ipynb`

* **Objective:**
  Use the descriptor CSV (from Example 2) to build and evaluate a regression model for property prediction.
* **What You Will Do:**

  * Upload descriptor CSV and (optionally) a property/target CSV or Excel.
  * Run cells to train a regression model (e.g., linear regression, random forest).
  * Visualize and interpret results (feature importance, predicted vs actual plot).

**How to Use:**

1. Open `regression_analysis.ipynb` in Colab.
2. Upload the required descriptor and target property files.
3. Execute the notebook step by step.
4. Review model evaluation metrics and plots generated.

---

## Example 4: Extending the Pipeline with Custom Features

**Path:** `examples/example4_custom/custom_feature_pipeline.ipynb`

* **Objective:**
  Learn to extend the pipeline by adding new feature extraction routines, then run the complete workflow.
* **What You Will Do:**

  * Modify or add Python code in the notebook to extract custom descriptors.
  * Run extraction and regression with your newly added features.

**How to Use:**

1. Open `custom_feature_pipeline.ipynb` in Colab.
2. Follow the instructions to define new feature extraction code blocks.
3. Upload your data, run all cells, and observe how the output changes.
4. Experiment by tweaking extraction logic or model parameters.

---

## FAQ and Troubleshooting

* **Missing Packages?**
  Run:

  ```python
  !pip install pandas numpy openpyxl scikit-learn matplotlib
  ```
* **File Upload Issues in Colab?**
  Use the `files.upload()` cell to interactively upload data files.
* **Data Format Problems?**
  Check the required input format as specified at the top of each notebook.
* **Need More Examples?**
  Copy and modify these notebooks for your own data.

---