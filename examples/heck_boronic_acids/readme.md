## Example 2: Batch Processing of Heck–Boronic Acid Derivatives

**Scenario:**  
You have a chemical library of Heck coupling products (from boronic acids) and want to rapidly extract all key DFT descriptors for machine learning, without manual feature engineering.

**Code Explanation:**  
- The notebook `heck_boronic_acids.ipynb` demonstrates:
  1. **Batch Upload**: Import a zipped folder or directory of log files representing a series of Heck coupling derivatives.
  2. **Batch Extraction**: Uses the batch processing function to extract descriptors from all molecules at once.
     ```python
     from extractor_regr import batch_extract_and_save
     batch_extract_and_save('/content/heck_boronic_logs/', '/content/heck_boronic_descriptors.csv')
     ```
     - The function automatically skips files with errors and reports any extraction issues for transparency.
  3. **Quick Analysis**: Loads the CSV and uses pandas to generate summary statistics (e.g., mean, std) of the extracted features.
  4. **Export**: Includes code to export the merged descriptor dataset for downstream ML tasks.

**Typical Output:**  
A large, merged descriptor matrix covering the full chemical library.

![Heck Boronic Acids Example](example.png)
