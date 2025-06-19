# 🏠 Housing Price Prediction - A Comprehensive Machine Learning Approach

This project builds a robust machine learning pipeline to predict housing prices based on structured data from the Kaggle House Prices dataset. It focuses on data preprocessing, feature engineering, and comparative evaluation of regression models to determine the most accurate solution.

---

## 📌 Objective

To develop a predictive system capable of estimating house sale prices using diverse numerical and categorical features such as size, quality, and location.

---

## 🧰 Tools & Technologies

- Python: pandas, numpy, matplotlib, seaborn, scikit-learn
- Jupyter Notebooks
- Dataset: Kaggle - House Prices: Advanced Regression Techniques

---

## 🗃️ Key Features

- **Data Cleaning**: Addressed missing values and removed outliers.
- **Feature Engineering**: Created new features like `TotalSF`, `TotalBathrooms`, and `HouseAge`, and applied log transformation to reduce skewness.
- **Encoding**: Applied one-hot encoding for categorical variables.
- **Modeling Techniques**:
  - Linear Regression
  - Ridge & Lasso Regression
  - Decision Tree & Random Forest
  - Gradient Boosting & XGBoost
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics**: R², MAE, MSE, RMSE

---

## 📊 Results Summary

| Model               | R² Score | RMSE   |
|--------------------|----------|--------|
| Random Forest       | 0.8866   | 29,490 |
| XGBoost             | 0.8819   | 30,093 |
| Ridge/Lasso/Linear  | ~0.8231  | ~36,836|
| Decision Tree       | 0.8074   | 38,434 |
| KNN                 | 0.7036   | 47,681 |
| SVR                 | -0.0246  | 88,652 |

✅ **Best Performing Model**: Random Forest Regressor

---

## 📂 Files Overview

- `Project Code.ipynb` — Jupyter notebook with the full modeling pipeline
- `Report.pdf` — Detailed report with methodology and results
- `housing_train.xlsx`, `housing_test.xlsx` — Dataset files
- `output_with_accuracy.xlsx` — Model predictions with performance scores
- `comparison.ipynb` — Comparative model performance notebook

---

## 🔍 Future Enhancements

- Handle multicollinearity using PCA or advanced feature selection
- Explore model stacking and ensemble learning
- Optimize model performance using advanced tuning methods like Bayesian optimization

---

## 👤 Authors

- Weiyu Chen  
- Rahul Muddhapuram  
- Alexis Myers  
- Sravanakumar Satish

---

## 📄 License

This project is made available **strictly for personal and educational use**.  
Redistribution, commercial use, or public publication of any portion is **not permitted** without prior permission from the authors.  
See `LICENSE_Educational_Only.txt` for full terms.

---

## 📬 Contact

For further details or collaboration inquiries, feel free to reach out via GitHub or LinkedIn.
