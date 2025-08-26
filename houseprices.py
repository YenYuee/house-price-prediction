import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df= pd.read_csv('train.csv')

#EDA
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())
print(df.describe())

#check correlation between features
#only numeric can be selected in heatmap
numeric_df= df.select_dtypes(include=['int64','float64'])
#top 10 most correlated features to SalePrice
print(numeric_df.corr()['SalePrice'].sort_values(ascending=False))

#boxplot for overallqual because it is categorical feature
# sns.boxplot(x=df['OverallQual'],y=df['SalePrice'])
# plt.show()

# #scatterplot for grlivearea because it is continous feature
# sns.scatterplot(x=df['GrLivArea'],y=df['SalePrice'])
# plt.show()

#check r2 score
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd' ,'FullBath']]
y=df['SalePrice']

#split the data to train and test to prevent overfitting
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
reg=LinearRegression().fit(X_train,y_train)

print('Train R²', reg.score(X_train,y_train))
print('Test R²', reg.score(X_test,y_test))

#Ridge
ridge_params={'alpha':[0.01,0.1,1,10,100]}
ridge_grid= GridSearchCV(Ridge(), ridge_params, cv=5,scoring='r2')
ridge_grid.fit(X_train,y_train)
print('Best Ridge alpha: ',ridge_grid.best_params_)
print('Best Ridge R²: ',ridge_grid.best_score_)

#Lasso
lasso_params={'alpha':[0.01,0.1,1,10,100]}
lasso_grid=GridSearchCV(Lasso(), lasso_params, cv=5, scoring='r2')
lasso_grid.fit(X_train,y_train)
print('Best Lasso alpha: ',lasso_grid.best_params_)
print('Best Lasso R² score: ',lasso_grid.best_score_)

#Linear Regression
lin=LinearRegression().fit(X_train,y_train)
print('Linear Regression R² score: ',lin.score(X_test,y_test))


#Prediction vs Actual
best_ridge=Ridge(alpha=ridge_grid.best_params_['alpha'])
best_ridge.fit(X_train,y_train)
y_pred= best_ridge.predict(X_test)
mse= mean_squared_error(y_test,y_pred)
rmse= np.sqrt(mse)
mape= np.mean(np.abs((y_test-y_pred)/y_test))*100
print('MSE: ', mse)
print("RMSE:", rmse)
print('MAPE', mape,"%")


#Visualize prediction vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Predicted vs Actual")
plt.show()
