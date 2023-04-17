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

######## Find correlation
cols = ["CalYear", "NumStationsWithPumpsAttending", "NumPumpsAttending", "PumpCount", "PumpHoursRoundUp","Notional Cost (£)"]
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
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