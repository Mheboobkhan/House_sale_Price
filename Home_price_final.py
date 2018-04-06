#Final model.....well i guess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#Importing the dataset 
dataset= pd.read_csv('train.csv')


#missing data
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
pd.read_csv('train.csv')
#dealing with missing data
dataset = dataset.drop((missing_data[missing_data['Total'] > 1]).index,1)
dataset = dataset.drop(dataset.loc[dataset['Electrical'].isnull()].index)
dataset.isnull().sum().max() #just checking that there's no missing data missing...

#Correilated Data 
#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(dataset[cols], size = 2.5)
plt.show();

dataset = pd.get_dummies(dataset)

#Creating the training set 
x= dataset[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','FullBath','TotRmsAbvGrd'
          ,'YearBuilt','YearRemodAdd']].values
y=dataset[['SalePrice']].values



#Data spliting 
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
y_test=y_test.astype(np.float64)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)


# Importing Test
Test = pd.read_csv('test.csv')

#missing data
total = Test.isnull().sum().sort_values(ascending=False)
percent = (Test.isnull().sum()/Test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)

#dealing with missing data
Test = Test.drop((missing_data[missing_data['Total'] > 1]).index,1)
Test['GarageCars'] = Test['GarageCars'].fillna("1")
Test['GarageArea'] = Test['GarageArea'].fillna("0")
Test['TotalBsmtSF'] = Test['TotalBsmtSF'].fillna("1288")



Test.isnull().sum().max() #just checking that there's no missing data missing...

x2= Test[['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','FullBath','TotRmsAbvGrd'
          ,'YearBuilt','YearRemodAdd']].values


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(x2)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


from sklearn.metrics import r2_score
accuracy =r2_score(y_test, y_pred2)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Prediction = lin_reg.predict(x2)

# Fitting the Random forest Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=1000,random_state=0)
regressor.fit(x,y)
# Predicting a new result
RF_reg = regressor.predict(x2)
accuracy =r2_score(y_test, RF_reg)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#make csv file 
submission = pd.DataFrame({ 'Id': x2[:,0],
                            'SalePrice': RF_reg })
submission.to_csv("submission2.csv", index=False)



