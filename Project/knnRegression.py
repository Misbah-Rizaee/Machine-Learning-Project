import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVR
from mlxtend.plotting import heatmap
from sklearn.neighbors import KNeighborsRegressor

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


######### Find correlation
#cols = ["CalYear", "NumStationsWithPumpsAttending", "NumPumpsAttending", "PumpCount", "PumpHoursRoundUp",
#        "Notional Cost (£)"]
#cm = np.corrcoef(df[cols].values.T)
#hm = heatmap(cm, row_names=cols, column_names=cols)
#plt.show()
# print(df.corr().to_string())


######### Lasso Regression model
X = df.iloc[:, :-1]  # Independent features
y = df.iloc[:, -1]  # Dependent feature

# To find optimum alpha parameter.
parameters = [7, 10, 50, 100, 200, 500, 1000]
rSquared = []
allMSE = []
allMAE = []

for i in parameters:
    knn_model = KNeighborsRegressor(n_neighbors=i)
    temp_rSquared = []
    temp_allMSE = []
    temp_allMAE = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        temp_rSquared.append(r2_score(y_test, y_pred))
        temp_allMSE.append(mean_squared_error(y_test, y_pred))
        temp_allMAE.append(mean_absolute_error(y_test, y_pred))
    rSquared.append(np.array(temp_rSquared).mean())
    allMSE.append(np.array(temp_allMSE).mean())
    allMAE.append(np.array(temp_allMAE).mean())

df_predictions = pd.DataFrame({'Alpha': parameters,
                               'r2_lasso': rSquared,
                               'MSE_lasso': allMSE,
                               'MAE_lasso': allMAE})
print(df_predictions)

# Plot the actual target value against the predicted target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsRegressor()  #0.001 is the most optimum alpha value
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

PumpCount_X_test = X_test.iloc[:, 4]
PumpHoursRoundUp_X_test = X_test.iloc[:, 5]

plt.scatter(PumpHoursRoundUp_X_test, y_test, marker='o', c='red')
plt.scatter(PumpHoursRoundUp_X_test, y_pred, marker='+', c='blue')
plt.title('Actual target value vs Predicted target value - Lasso regression')
plt.xlabel('PumpHoursRoundUp', fontsize=14)
plt.ylabel('Notional Cost', fontsize=14)
plt.legend(("Actual target value", "Predicted target value"))
plt.show()

plt.scatter(PumpCount_X_test, y_test, marker='o', c='red')
plt.scatter(PumpCount_X_test, y_pred, marker='+', c='blue')
plt.title('Actual target value vs Predicted target value - Lasso regression')
plt.xlabel('PumpCount', fontsize=14)
plt.ylabel('Notional Cost', fontsize=14)
plt.legend(("Actual target value", "Predicted target value"))
plt.show()

######### Plot (Cost vs PumpHoursRoundUp) graph
PumpCount = df.iloc[:, 4]
PumpHoursRoundUp = df.iloc[:, 5]

plt.scatter(PumpHoursRoundUp, y, marker='+', c='blue')
plt.title('Cost vs PumpHoursRoundUp - Lasso regression')
plt.xlabel('PumpHoursRoundUp')
plt.ylabel('Notional Cost (£)')
plt.show()

######### Plot (Cost vs PumpHoursRoundUp) graph
plt.scatter(PumpCount, y, marker='+', c='blue')
plt.title('Cost vs PumpCount - Lasso regression')
plt.xlabel('PumpCount')
plt.ylabel('Notional Cost (£)')
plt.show()