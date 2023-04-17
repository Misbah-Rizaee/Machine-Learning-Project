import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mlxtend.plotting import heatmap

df = pd.read_csv('LFB_edit.csv', encoding='unicode_escape')
# print(df)


######### Get the first 5000 rows for each year
df_2017 = df[df['CalYear'] >= 2017].head(5000)
df_2018 = df[df['CalYear'] >= 2018].head(5000)
df_2019 = df[df['CalYear'] >= 2019].head(5000)
df_2020 = df[df['CalYear'] >= 2020].head(5000)
df_2021 = df[df['CalYear'] >= 2021].head(5000)
frames = [df_2017, df_2018, df_2019, df_2020, df_2021]
df = pd.concat(frames)
# print(df)



######### Check Null values
df['NumStationsWithPumpsAttending'].fillna(method='ffill', inplace=True, axis=0)
df['NumPumpsAttending'].fillna(method='ffill', inplace=True, axis=0)
df['PumpCount'].fillna(method='ffill', inplace=True, axis=0)
df['PumpHoursRoundUp'].fillna(method='ffill', inplace=True, axis=0)
df['Notional Cost (£)'].fillna(method='ffill', inplace=True, axis=0)
# print(df.isnull().sum())



######### Converting the labels into a numeric
le = LabelEncoder()
df['CalYear'] = le.fit_transform(df['CalYear'])
df['IncidentGroup'] = le.fit_transform(df['IncidentGroup'])
df['PropertyCategory'] = le.fit_transform(df['PropertyCategory'])
df['PropertyType'] = le.fit_transform(df['PropertyType'])
df['Notional Cost (£)'] = le.fit_transform(df['Notional Cost (£)'])
# print(df)



######### Normalize
def norm(df):
    for i in range(0, len(df.columns)):
        min_ = df.iloc[:, i].min(axis=0)
        max_ = df.iloc[:, i].max(axis=0)
        # print(min_, " ", max_)
        df.iloc[:, i] = (df.iloc[:, i] - min_) / (max_ - min_)
    return df
norm(df)
print(df.to_string())

# Normalize 2
# def norm(df):
#     for i in range(0, len(df.columns)):
#         std = df.iloc[:, i].std()
#         mean = df.iloc[:, i].mean()
#         df.iloc[:, i] = (df.iloc[:, i] - mean) / std
#     return df
# norm(df)
# print(df.to_string())



######### Find correlation
cols = ["CalYear", "NumStationsWithPumpsAttending", "NumPumpsAttending", "PumpCount", "PumpHoursRoundUp",
        "Notional Cost (£)"]
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()
# print(df.corr().to_string())




######### Ridge Regression model
X = df.iloc[:, :-1]  # Independent features
y = df.iloc[:, -1]  # Dependent feature


# To find optimum alpha parameter.
parameters = [1e-3, 1e-2, 1, 10, 50, 100, 500, 1000]
rSquared = []
allMSE = []
allMAE = []

for i in parameters:
    ridge_model = Ridge(alpha=i)
    # ridge_model.fit(X_train, y_train)
    # y_pred = ridge_model.predict(X_test)

    temp_rSquared = []
    temp_allMSE = []
    temp_allMAE = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # print(len(X_train))
        # print(len(y_train))
        # ridge_model.fit(X[train], y[train])
        # y_pred = ridge_model.predict(X[test])
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)

        temp_rSquared.append(r2_score(y_test, y_pred))
        temp_allMSE.append(mean_squared_error(y_test, y_pred))
        temp_allMAE.append(mean_absolute_error(y_test, y_pred))

    rSquared.append(np.array(temp_rSquared).mean())
    allMSE.append(np.array(temp_allMSE).mean())
    allMAE.append(np.array(temp_allMAE).mean())

df_predictions = pd.DataFrame({'Alpha': parameters,
                               'r2_ridge': rSquared,
                               'MES_ridge': allMSE,
                               'MAE_ridge': allMAE})
print(df_predictions)



# Plot the actual target value against the predicted target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Ridge(alpha=1) # 1 is the most optimum alpha value
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

PumpCount_X_test = X_test.iloc[:, 7]
PumpHoursRoundUp_X_test = X_test.iloc[:, 8]

plt.scatter(PumpHoursRoundUp_X_test, y_test, marker='+', c='red')
plt.scatter(PumpHoursRoundUp_X_test, y_pred, marker='+', c='blue')
plt.title('Actual target value vs Predicted target value - Ridge regression')
plt.xlabel('PumpHoursRoundUp', fontsize=14)
plt.ylabel('Notional Cost', fontsize=14)
plt.legend(("Actual target value", "Predicted target value"))
plt.show()



######### Plot (Cost vs PumpHoursRoundUp) graph
CalYear = df.iloc[:, 0]
PumpCount = df.iloc[:, 7]
PumpHoursRoundUp = df.iloc[:, 8]

plt.scatter(PumpHoursRoundUp, y, marker='+', c='blue')
plt.title('Cost vs PumpHoursRoundUp - Ridge regression')
plt.xlabel('PumpHoursRoundUp')
plt.ylabel('Notional Cost (£)')
plt.show()

######### Plot (Cost vs PumpHoursRoundUp) graph
plt.scatter(PumpCount, y, marker='+', c='blue')
plt.title('Cost vs PumpCount - Ridge regression')
plt.xlabel('PumpCount')
plt.ylabel('Notional Cost (£)')
plt.show()




