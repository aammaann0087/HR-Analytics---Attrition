---
Title: "HR-Analytics---Attrition"
Author: "Aman"
Date: "07/03/2025"
---




**1. Introduction**

This report details the process of analyzing employee attrition using logistic regression. The dataset contains various employee attributes, and our goal is to identify key factors influencing attrition. We perform feature selection using statistical tests and handle class imbalance to improve model performance.
________________________________________

**2. Data Preprocessing**

**2.1 Importing Libraries**

We import necessary libraries for data manipulation, visualization, and model building:

•	pandas, numpy - for data processing

•	seaborn, matplotlib.pyplot - for visualization

•	sklearn.model_selection - for splitting data

•	sklearn.preprocessing - for scaling and encoding data

•	statsmodels.api, scipy.stats - for statistical testing

•	imblearn.over_sampling.SMOTE - for handling class imbalance

•	sklearn.metrics - for evaluating model performance

•	sklearn.linear_model.LogisticRegression - for training the model

**2.2 Loading the Data**

`df = pd.read_csv('HR_Analytics.csv')`

This loads the dataset into a Pandas DataFrame for further processing.

**2.3 Data Cleaning**

•	Handling Missing Values: `df = df.dropna()` removes all rows with missing values.

•	Retaining Numeric and Categorical Data: `df.select_dtypes(include=['number', 'object'])` ensures only relevant data types are used.

•	Converting Categorical Variables: `df[col] = df[col].astype('category')` ensures categorical columns are treated correctly.

•	Encoding Target Variable: The target variable Attrition is mapped to binary values (Yes = 1, No = 0).

**2.4 Visualizing Class Imbalance**

A count plot is created to show the distribution of employees who left versus those who stayed.

`sns.countplot(x=y, palette='coolwarm')`

This helps us determine whether resampling is needed to balance the dataset.
________________________________________

**3. Feature Selection**

**3.1 Chi-Square Test for Categorical Variables**

A chi-square test is conducted to check the relationship between categorical variables and attrition. Only features with p < 0.05 are selected.

**3.2 ANOVA Test for Numerical Variables**

ANOVA is used to check if numerical variables significantly differ between attrition groups. Variables with p < 0.05 are selected.

**3.3 Removing Highly Correlated Features**

To prevent multicollinearity, highly correlated features (|correlation| > 0.9) are dropped.

`corr_matrix = df_selected.corr()`

A correlation heatmap is plotted to visualize feature relationships.
________________________________________

**4. Data Preparation for Modeling**

**4.1 Encoding Categorical Variables**

•	Binary Variables: Encoded using `LabelEncoder()`.

•	Multi-Class Variables: Converted to dummy variables using `pd.get_dummies()`.

**4.2 Train-Test Split**

The dataset is split into training (80%) and testing (20%) subsets:

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

**4.3 Handling Class Imbalance with SMOTE**

Synthetic Minority Oversampling (SMOTE) is applied to balance the dataset.

`smote = SMOTE(random_state=42)`

`X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)`

**4.4 Feature Scaling**

Standardization is applied to ensure all numerical features have similar distributions.

`scaler = StandardScaler()`

`X_train_scaled = scaler.fit_transform(X_train_sm)`

`X_test_scaled = scaler.transform(X_test)`
________________________________________

**5. Model Training and Evaluation**

**5.1 Logistic Regression Model**

The logistic regression model is trained on the processed data:

`logit_model = LogisticRegression()`

`logit_model.fit(X_train_scaled, y_train_sm)`

**5.2 Model Evaluation**

Accuracy, Confusion Matrix, and Classification Report

`print("Accuracy:", accuracy_score(y_test, y_pred))`

`print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))`

`print("Classification Report:\n", classification_report(y_test, y_pred))`

These metrics help assess model performance:

•	Accuracy: Measures overall correctness.

•	Confusion Matrix: Shows the number of correct and incorrect classifications.

•	Precision, Recall, and F1-score: Evaluate performance on each class.

After training the logistic regression model, we evaluated its performance on the test dataset. Below are the results:

- **Accuracy Score**

The model achieved an accuracy of 88.07%, indicating a high overall correctness in predicting attrition.

- **Confusion Matrix**

  |228  9|

  |25  23|

True Negatives (TN) = 228 → Employees correctly predicted to stay.

False Positives (FP) = 9 → Employees incorrectly predicted to leave.

False Negatives (FN) = 25 → Employees incorrectly predicted to stay.

True Positives (TP) = 23 → Employees correctly predicted to leave.

- **Classification Report**

| Class              | Precision | Recall | F1-score | Support |
|--------------------|-----------|--------|----------|---------|
| No Attrition (0)  | 0.90      | 0.96   | 0.93     | 237     |
| Attrition (1)     | 0.72      | 0.48   | 0.57     | 48      |
| **Macro Avg**     | 0.81      | 0.72   | 0.75     | 285     |
| **Weighted Avg**  | 0.87      | 0.88   | 0.87     | 285     |

Precision (0.72) for attrition means that 72% of employees predicted to leave actually left.

Recall (0.48) for attrition is relatively low, meaning the model missed 52% of actual attrition cases.

F1-score for attrition is 0.57, indicating room for improvement in correctly identifying employees likely to leave.

- **Insights**

The model performs well at predicting employees who will stay (96% recall for "No Attrition").

However, it struggles to identify all employees who will leave (48% recall for "Attrition").

The precision for attrition cases is 72%, meaning when the model predicts an employee will leave, it is correct in most cases.

To improve recall for attrition, we could explore additional feature engineering, tuning the model hyperparameters, or using a different classification threshold.

**5.3 ROC Curve**

The ROC curve visualizes the tradeoff between true positives and false positives.

`fpr, tpr, _ = roc_curve(y_test, y_pred_prob)`

`roc_auc = auc(fpr, tpr)`

AUC (Area Under Curve) measures model performance; higher values indicate better discrimination.
________________________________________

**6. Conclusion and Insights**

- AUC Score (0.87) indicates strong model performance.

- The model effectively distinguishes between attrition and non-attrition cases.

- A good balance between True Positive Rate and False Positive Rate is observed.

- Some attrition cases are missed, suggesting recall improvement is needed.

- Adjusting the classification threshold can optimize precision-recall trade-off.

- The model is reliable but can be fine-tuned for better performance.

This report summarizes the entire process from data preprocessing to model evaluation, highlighting key methodologies used in HR Analytics for employee attrition prediction.
