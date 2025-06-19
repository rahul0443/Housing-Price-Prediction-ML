# Housing Price Prediction - A Comprehensive Machine Learning Approach

This project builds a robust machine learning pipeline to predict housing prices based on structured data from the Kaggle House Prices dataset. It focuses on data preprocessing, feature engineering, and the comparative evaluation of several regression models to determine the most accurate approach.

---

## 📌 Objective

To develop a predictive system capable of estimating house sale prices using diverse numerical and categorical features such as size, quality, and location.

---

## 🧰 Tools & Technologies

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
- Jupyter Notebooks
- Kaggle Dataset (House Prices: Advanced Regression Techniques)

---

## 🗃️ Key Features

- **Data Cleaning**: Handled missing values and outliers.
- **Feature Engineering**: Created `TotalSF`, `TotalBathrooms`, `HouseAge`, and applied log transformation to skewed features.
- **Encoding**: One-hot encoding for categorical variables.
- **Modeling**: 
  - Linear Regression
  - Ridge & Lasso Regression
  - Decision Tree & Random Forest
  - Gradient Boosting & XGBoost
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics**: R², MAE, MSE, RMSE

---

## 📊 Results Summary

| Model                 | R² Score | RMSE     |
|----------------------|----------|----------|
| Random Forest        | 0.8866   | 29,490   |
| XGBoost              | 0.8819   | 30,093   |
| Ridge/Lasso/Linear   | ~0.8231  | ~36,836  |
| Decision Tree        | 0.8074   | 38,434   |
| KNN                  | 0.7036   | 47,681   |
| SVR                  | -0.0246  | 88,652   |

📌 **Best Performing Model**: Random Forest Regressor

---

## 📂 Files Overview

- `Project Code.ipynb` — Jupyter notebook containing full pipeline.
- `Report.pdf` — Detailed report with methodology and results.
- `housing_train.xlsx`, `housing_test.xlsx` — Datasets used for modeling.
- `output_with_accuracy.xlsx` — Prediction outputs with performance metrics.
- `comparison.ipynb` — Comparative model analysis.

---

## 🔍 Future Enhancements

- Address multicollinearity using PCA or feature selection.
- Experiment with stacking ensemble models.
- Improve hyperparameter optimization with Bayesian search.

---

## 👤 Authors

Group 06  
- Weiyu Chen  
- Rahul Muddhapuram  
- Alexis Myers  
- Sravanakumar Satish

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📎 Acknowledgments

Thanks to the Kaggle competition organizers and CSE 572 instructors for guidance and support.
