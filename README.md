🌍 Human Development & Happiness Index – Machine Learning Project

This project builds and compares multiple machine learning models to analyze and predict:

Human Development Index (HDI) (regression problem)

Happiness Index (classification problem)

The goal is to understand which socio-economic factors influence development and happiness and to compare the performance of different ML models.

📌 Project Overview
Models Built
Index	Type	Models Used
HDI Index	Regression	Linear Regression, Random Forest, Gradient Boosting
Happiness Index	Classification	Logistic Regression, Random Forest, Gradient Boosting

📊 HDI INDEX MODEL
🎯 Objective

Predict Human Development Index (HDI) using socio-economic indicators such as:

Life expectancy

Education levels

Trade & economy indicators

Happiness indicators

🧠 Models Used

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

📈 Evaluation Metrics

MAE (Mean Absolute Error)

R² Score

Cross-Validation R²

🏆 Best Performing Model

Random Forest Regressor

Test R² ≈ 0.76

CV R² ≈ 0.60

😊 HAPPINESS INDEX MODEL
🎯 Objective

Classify countries into Happiness Categories:

Happy

Very Happy

Unhappy

🧠 Models Used

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

📈 Evaluation Metrics

Accuracy

F1-score (macro)

Cross-validation F1-score

🏆 Best Performing Model

Random Forest Classifier

High F1-score

Strong generalization across folds

🔍 Feature Importance Analysis

Both models include feature importance analysis to identify the most influential factors.

Key Insights:

Happiness level strongly influences HDI

Life expectancy & education are major contributors

Economic and innovation indicators also matter

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone 
cd HDI-Happiness-ML

2️⃣ Create Virtual Environment (Recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ How to Run the Models
Run HDI Model
python HDI.py

Run Happiness Model
python happiness_index.py

🧪 Libraries Used

Python 3.10+

pandas

numpy

scikit-learn

matplotlib

📌 Key Learnings

Tree-based models outperform linear models on complex socio-economic data

Feature scaling is crucial for linear classifiers

Cross-validation prevents misleading performance metrics

Happiness and development are strongly interconnected

🚀 Future Improvements

Add XGBoost / LightGBM

Hyperparameter tuning (GridSearchCV)

SHAP-based explainability

Interactive dashboards (Streamlit)

👤 Author

Developed for a hackathon from SOIL institute of India

Machine Learning

Socio-economic indicators

Model comparison & explainability
