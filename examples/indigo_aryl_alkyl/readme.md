## Example 3: Regression Modeling for Indigo Aryl–Alkyl Derivatives

**Scenario:**  
You are interested in predicting physicochemical or spectroscopic properties (e.g., UV/Vis absorption) of indigo aryl–alkyl derivatives based on quantum chemical descriptors.

**Code Explanation:**  
- The notebook `indigo_aryl_alkyl.ipynb` walks you through:
  1. **Input Preparation**: Upload both descriptor CSV (from previous extraction) and an Excel file containing experimental or calculated property values.
  2. **Data Merge**: Merges the descriptors and property labels into a single DataFrame, aligning by molecule ID.
  3. **Regression Pipeline**:  
     - Splits the data into training/test sets.
     - Trains a regression model (e.g., RandomForestRegressor or LinearRegression) to predict target properties.
     - Evaluates performance using metrics such as MAE (Mean Absolute Error), R² score, and cross-validation.
     ```python
     from regressor import run_regression_analysis
     run_regression_analysis('/content/indigo_descriptors.csv')
     ```
  4. **Visualization**:  
     - Plots predicted vs. actual property values for test molecules.
     - Ranks descriptor importance for interpretation.

**Typical Output:**  
Graphs and metrics summarizing model accuracy and feature importance, plus a ready-to-use ML pipeline.

![Indigo Aryl Alkyl Example](example.png)
