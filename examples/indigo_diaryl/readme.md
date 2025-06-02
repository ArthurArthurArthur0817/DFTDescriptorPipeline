## Example 4: Pipeline Customization – Indigo Diaryl Series

**Scenario:**  
You need to add custom descriptor extraction (e.g., a special property unique to diaryl indigo compounds), and want to see how new features impact model performance.

**Code Explanation:**  
- The notebook `indigo_diaryl.ipynb` demonstrates:
  1. **Custom Extraction**:  
     - Shows how to modify or extend the pipeline to extract new quantum descriptors (for example, a unique orbital property or symmetry function).
     - Example: Adding code to extract a custom value from each log file and append it to the dataset.
     ```python
     # Example of adding a new feature extraction
     def extract_custom_property(log_file):
         # Your logic here
         return custom_value
     ```
  2. **Full Workflow**:  
     - Runs extraction with both standard and custom features.
     - Retrains regression models using the updated dataset.
     - Compares model performance before and after the addition of custom features.
  3. **Results Analysis**:  
     - Interprets whether the new feature improves predictions.
     - Provides code for feature selection and advanced visualization.

**Typical Output:**  
An updated descriptor table with new custom columns, and model evaluation results showing the effect of feature engineering.

![Indigo Diaryl Example](example.png)
