import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('Book3.csv', encoding='unicode_escape')



######### Check Null values
df['NumStationsWithPumpsAttending'].fillna(method='ffill', inplace=True, axis=0)
df['NumPumpsAttending'].fillna(method='ffill', inplace=True, axis=0)
df['PumpCount'].fillna(method='ffill', inplace=True, axis=0)
df['PumpHoursRoundUp'].fillna(method='ffill', inplace=True, axis=0)
# print(df.isnull().sum())


######### Converting the labels into a numeric
le = LabelEncoder()
df['IncidentGroup'] = le.fit_transform(df['IncidentGroup'])


######### Normalize
def norm(df):
    for i in range(0, len(df.columns)):
        min_ = df.iloc[:, i].min(axis=0)
        max_ = df.iloc[:, i].max(axis=0)
        df.iloc[:, i] = (df.iloc[:, i] - min_) / (max_ - min_)
    return df

def norm2(df):
	for i in range(0, len(df.columns)):
		if(i != 1):
			std = df.iloc[:,i].std()
			mean = df.iloc[:,i].mean()
			df.iloc[:,i] = (df.iloc[:,i] - mean) / std
	return df

def norm3(df):
    for i in range(0, len(df.columns)):
        max_ = df.iloc[:, i].max(axis=0)
        df.iloc[:, i] = (df.iloc[:, i]) / (max_)
    return df

df = norm3(df)

######### Lasso Regression model
X = df.iloc[:, :-1]  # Independent features
y = df.iloc[:, -1]  # Dependent feature

# To find optimum alpha parameter.
# To find optimum alpha parameter.
parameters = [1e-3, 1e-2, 1, 10, 100, 1000]
rSquared = []
allMSE = []
allMAE = []
std_error = []
for i in parameters:
    model = Lasso(alpha=i)
    temp_rSquared = []
    temp_allMSE = []
    temp_allMAE = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        temp_rSquared.append(r2_score(y_test, y_pred))
        temp_allMSE.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        temp_allMAE.append(mean_absolute_error(y_test, y_pred))
    rSquared.append(np.array(temp_rSquared).mean())
    allMSE.append(np.array(temp_allMSE).mean())
    allMAE.append(np.array(temp_allMAE).mean())
    std_error.append(np.array(temp_allMSE).std())
plt.title("RMSE against alpha")
plt.errorbar(parameters,allMSE,yerr=std_error,linewidth=2)
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.legend()
plt.show()

df_predictions = pd.DataFrame({'Alpha': parameters,
                               'r2_lasso': rSquared,
                               'MSE_lasso': allMSE,
                               'MAE_lasso': allMAE})
print(df_predictions)

# Plot the actual target value against the predicted target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Lasso(0.01)  #0.001 is the most optimum alpha value
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MEAN SQURED ERROR")
print(math.sqrt(mean_squared_error(y_test,y_pred)))
print("MEAN ABSOLUTE ERROR")
print(mean_absolute_error(y_test,y_pred))
print("R2 SCORE")
print(r2_score(y_test,y_pred))

PumpCount_X_test = X_test.iloc[:, 4]
PumpHoursRoundUp_X_test = X_test.iloc[:, 5]

PumpCount_X_train = X_train.iloc[:, 4]
PumpHoursRoundUp_X_train = X_train.iloc[:, 5]

plt.scatter(PumpHoursRoundUp_X_train, y_train, marker='o', c='red')
plt.scatter(PumpHoursRoundUp_X_test, y_pred, marker='+', c='blue')
plt.title('Actual target value vs Predicted target value - Lasso regression')
plt.xlabel('PumpHoursRoundUp', fontsize=14)
plt.ylabel('Notional Cost', fontsize=14)
plt.legend(("Actual target value", "Predicted target value"))
plt.show()

plt.scatter(PumpCount_X_train, y_train, marker='o', c='red')
plt.scatter(PumpCount_X_test, y_pred, marker='+', c='blue')
plt.title('Actual target value vs Predicted target value - Lasso regression')
plt.xlabel('PumpCount', fontsize=14)
plt.ylabel('Notional Cost', fontsize=14)
plt.legend(("Actual target value", "Predicted target value"))
plt.show()