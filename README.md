# Insurance Cost Prediction

This is a **Machine Learning** project aimed at predicting health insurance costs based on various demographic and health-related features using regression models.

### **Project Overview:**
The project involves building and evaluating multiple machine learning models to predict health insurance charges for individuals. The dataset includes various features like age, gender, BMI, smoking status, number of children, and region, which are used to predict the **charges** (target variable). The project explores several models, from **Multiple Linear Regression (MLR)** to more complex models like **Random Forest Regression (RFR)**, with techniques like **Principal Component Analysis (PCA)** for dimensionality reduction.

### **Columns in the Dataset:**
- **age**: Age of the primary beneficiary
- **sex**: Gender of the insurance contractor (female, male)
- **bmi**: Body Mass Index (BMI) – A measure of body fat based on height and weight (kg/m²)
- **children**: Number of children/dependents covered by the health insurance
- **smoker**: Whether the individual is a smoker (yes, no)
- **region**: The region where the beneficiary resides in the US (northeast, southeast, southwest, northwest)
- **charges**: The individual medical costs billed by health insurance (target variable)

### **Key Models Implemented:**

1. **MLR_before_outlier**: 
   - **Multiple Linear Regression (MLR)** model applied to the dataset without handling outliers.
   - **RMSE** and **accuracy** were calculated as performance metrics.
   
2. **MLR_after_outlier**: 
   - **Multiple Linear Regression (MLR)** model applied after removing outliers from the dataset. 
   - Outlier removal resulted in improved model performance, particularly in terms of **RMSE**.

3. **RFR_after_outlier**:
   - **Random Forest Regression (RFR)** model applied after removing outliers.
   - This model also showed improvements in prediction accuracy compared to MLR.

4. **MLR with PCA**: 
   - **Principal Component Analysis (PCA)** applied to reduce the dimensionality of the dataset before fitting the **Multiple Linear Regression (MLR)** model.
   - PCA helped simplify the model but resulted in a slight drop in performance compared to the outlier-free MLR model.

5. **RFR**: 
   - **Random Forest Regression (RFR)** applied to the dataset, without any outlier handling or dimensionality reduction.
   - This model was one of the best-performing models, achieving high **accuracy** and low **RMSE**.

6. **RFR_with_hyper_parameter_tuning**:
   - **Hyperparameter tuning** of the **Random Forest Regression (RFR)** model using **GridSearchCV** to optimize model performance.
   - This model achieved the best performance, with the highest **accuracy** and the lowest **RMSE**.

7. **RFR with PCA**:
   - **Random Forest Regression (RFR)** applied after performing **Principal Component Analysis (PCA)** for dimensionality reduction.
   - This model showed acceptable performance but did not outperform the hyperparameter-tuned RFR model.

### **Exploratory Data Analysis (EDA) & Visualizations:**

#### **1. Dataset Overview and Inspection:**
   - **Loading the Dataset**: The dataset was loaded and inspected using basic commands (`df.head()`, `df.info()`) to understand the structure and dimensions (rows and columns).
   - **Missing Data**: Checked for missing values using `df.isnull().sum()` to assess if data imputation or cleaning was needed.

#### **2. Descriptive Statistics:**
   - **Statistical Summary**: Generated summary statistics using `df.describe()` to analyze the distribution of numerical variables (mean, median, standard deviation, min/max values).
   - **Data Types**: Identified the data types of each column (categorical vs. numerical).

#### **3. Data Visualization:**
   - **Histograms**: Plotted histograms for numerical features like `age`, `bmi`, `charges`, and `children` to visualize their distributions and detect skewness or abnormal data points.
   - **Boxplots**: Used boxplots to identify and visualize potential **outliers** in numerical features like `bmi` and `charges`.

#### **4. Correlation Analysis:**
   - **Correlation Heatmap**: Generated a heatmap using `seaborn` to visualize correlations between numerical features. This helped understand relationships between variables like `bmi` and `charges`.
   - **Scatter Plots**: Plotted scatter plots to examine the relationship between individual variables, particularly between `bmi` and `charges`, as this is an important predictor of health insurance cost.

#### **5. Categorical Feature Analysis:**
   - **Bar Plots**: Plotted bar charts for categorical features like `sex`, `smoker`, and `region` to examine their distribution and frequencies.
   - **Group-by Analysis**: Used `groupby()` to calculate the mean insurance charges for different categories (e.g., average charges by smoking status, region, etc.).

#### **6. Outlier Detection & Removal:**
   - **Boxplots for Outliers**: Used boxplots to detect and visualize outliers in numerical columns like `charges`. Outliers were removed or capped to reduce their influence on model performance.
   - **Outlier Impact**: Analyzed the effect of removing outliers on model performance, particularly on **RMSE** and **accuracy**.

#### **7. Feature Engineering:**
   - **Encoding Categorical Variables**: Categorical columns (`sex`, `smoker`, and `region`) were encoded using **one-hot encoding** to convert them into numerical values suitable for machine learning models.
   - **Feature Scaling**: Applied **StandardScaler** to scale numerical features like `bmi` and `charges`, which helps models like **Linear Regression** converge faster.

#### **8. Pairwise Relationships:**
   - **Pairplot**: Used a pairplot to visualize pairwise relationships among numerical variables. This helped assess how features like `bmi`, `age`, and `charges` correlate with each other.
   - **Feature Importance**: Ran **Random Forest** to identify which features (e.g., `age`, `bmi`, `smoker`) were the most important in predicting insurance charges.

#### **9. Data Preprocessing for Modeling:**
   - **Train-Test Split**: Split the dataset into training and test sets using `train_test_split()` to ensure model evaluation on unseen data.
   - **Dimensionality Reduction (PCA)**: Applied **PCA** to reduce the number of features, simplifying the model while retaining most of the variance in the data.

### **Results:**
- **Best Model**: The **Random Forest Regression (RFR)** model with hyperparameter tuning achieved the highest **accuracy (85.1%)** and lowest **RMSE (5007.81)**.
- **Impact of Outliers**: Removing outliers significantly improved the performance of models like **MLR** and **RFR**.
- **PCA**: While PCA helped reduce dimensionality, it caused a slight performance trade-off compared to using the full feature set.

### **Technologies and Libraries Used:**
- **Python** for data manipulation and model building.
- **Pandas** and **NumPy** for data handling.
- **Scikit-learn** for machine learning algorithms and evaluation.
- **Matplotlib** and **Seaborn** for visualizations.

### **Usage:**
To run the project, clone this repository and execute the insurance_cost_prediction.ipynb notebook to see the data processing, model implementation, and evaluation results.
