# 🏡 House Price Prediction Project  

A machine learning project to predict house prices using regression techniques.  
The dataset is based on the **Kaggle House Prices - Advanced Regression Techniques** competition.  

---

## 📊 Project Overview  
This project applies **Exploratory Data Analysis (EDA)** and **Machine Learning models** to predict house prices.  
I experimented with **Linear Regression, Ridge, and Lasso Regression** to improve performance and reduce overfitting.  

---

## ⚙️ Technologies Used  
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔬 Exploratory Data Analysis (EDA)  
- Checked dataset shape, info, and missing values  
- Examined feature correlations with `SalePrice`  
- Visualized categorical & continuous features (boxplots & scatterplots)  

---

## 🧠 Models Used  
- **Linear Regression**  
- **Ridge Regression** (with hyperparameter tuning using GridSearchCV)  
- **Lasso Regression** (with hyperparameter tuning using GridSearchCV)  

---

## 📈 Results  

- **Train R²**: `0.7508`  
- **Test R²**: `0.7924`  
- **Best Ridge alpha**: `10`  
- **Best Ridge R² (CV)**: `0.7339`  
- **Best Lasso alpha**: `100`  
- **Best Lasso R² (CV)**: `0.7339`  
- **Linear Regression Test R²**: `0.7924`  

**Error Metrics (Best Ridge):**  
- **MSE**: `1,590,310,548.93`  
- **RMSE**: `39,878.70`  
- **MAPE**: `15.55%`  

---

## 📊 Visualization  

**Predicted vs Actual SalePrice**  
The scatter plot shows how close the predicted values are to the actual values.  
A strong diagonal line indicates good predictions.  

![Prediction vs Actual](prediction_vs_actual.png)  

---

## 🚀 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
