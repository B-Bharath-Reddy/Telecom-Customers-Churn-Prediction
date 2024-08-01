# Telecom Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecom company using various machine learning models. Customer churn refers to the loss of clients or customers, and predicting churn helps businesses take proactive measures to retain customers, thereby reducing churn rates and increasing revenue.

## Dataset
The dataset used in this project contains customer information including demographic details, account information, and usage patterns. Each record is labeled with whether the customer has churned or not.

### Key Features
- **Customer ID**: Unique identifier for each customer.
- **Demographics**: Age, gender, and other personal information.
- **Account Information**: Contract type, tenure, monthly charges, etc.
- **Usage Patterns**: Data usage, call duration, etc.
- **Churn**: Whether the customer has churned (Yes/No).

## Project Structure
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Understanding the distribution and relationships within the data.
- **Model Training and Evaluation**: Training multiple machine learning models and evaluating their performance using various metrics.
- **Feature Importance**: Identifying the most significant features affecting customer churn.

## Models Evaluated
Several machine learning models were trained and evaluated to predict customer churn:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- XGBoost Classifier

## Performance Metrics
The models were evaluated based on the following metrics:
- F1 Score
- Precision
- Recall
- ROC-AUC Score

## Results
After evaluating multiple models, the following key insights were derived:

### Logistic Regression
- **Test ROC-AUC Score**: 85.59%
- **Test F1 Score**: 78.62%
- Demonstrated balanced performance with moderate precision and recall.

### Decision Tree Classifier
- **Test ROC-AUC Score**: 86.04%
- **Test F1 Score**: 80.15%
- Showed slightly better performance than logistic regression, but with some risk of overfitting.

### Random Forest Classifier
- **Train ROC-AUC Score**: 99.99%
- **Test ROC-AUC Score**: 92.11%
- **Test F1 Score**: 85.24%
- The model showed excellent performance, indicating its robustness in predicting churn.

### Gradient Boosting Classifier
- **Test ROC-AUC Score**: 92.30%
- **Test F1 Score**: 83.77%
- Demonstrated strong performance with balanced precision and recall, slightly lower than Random Forest.

### AdaBoost Classifier
- **Test ROC-AUC Score**: 91.53%
- **Test F1 Score**: 83.09%
- Performed well, showing strong generalization capability.

### XGBoost Classifier
- **Test ROC-AUC Score**: 92.41%
- **Test F1 Score**: 84.52%
- This model showed the best overall performance among all models, with balanced precision and recall.

## Recommendation
Based on the evaluation metrics, the **XGBoost Classifier** is recommended for predicting telecom customer churn due to its superior performance in terms of test ROC-AUC score (92.41%) and balanced F1 score (84.52%). This model effectively balances the trade-off between precision and recall, making it a reliable choice for identifying customers at risk of churning.

## Business Impact
Implementing the XGBoost churn prediction model can significantly benefit the telecom company by:
- Identifying at-risk customers with high accuracy, allowing for targeted retention strategies.
- Reducing customer churn rates and increasing customer loyalty.
- Enhancing revenue by focusing on customer retention rather than acquisition.

## Future Work
To further improve the model's performance and ensure its continued effectiveness, the following steps are recommended:
- **Hyperparameter Tuning**: Further fine-tuning of model hyperparameters can enhance predictive performance.
- **Feature Engineering**: Adding more relevant features or data sources could provide better insights and improve model accuracy.
- **Model Monitoring**: Regularly updating and monitoring the model to adapt to changing customer behaviors and market conditions.
