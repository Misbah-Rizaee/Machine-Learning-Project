import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split


def norm(df):
	for i in range(0, len(df.columns)):
		min_ = df.iloc[:,i].min(axis=0)
		max_ = df.iloc[:,i].max(axis=0)
		print(min_, " ", max_)
		df.iloc[:,i] = (df.iloc[:,i] - min_) / (max_ - min_)
	return df
	

df = pd.read_csv('Book1.csv')
print(df)
#print(df.size)
#print(df)
#print(df.isnull().sum())
df['NumStationsWithPumpsAttending'].fillna(method='ffill',inplace=True,axis=0)
df['NumPumpsAttending'].fillna(method='ffill',inplace=True,axis=0)
df['PumpCount'].fillna(method='ffill',inplace=True,axis=0)
df['PumpHoursRoundUp'].fillna(method='ffill',inplace=True,axis=0)
df['Notional Cost (£)'].fillna(method='ffill',inplace=True,axis=0)
#print(df.isnull().sum())
le = LabelEncoder()
df['CalYear']=le.fit_transform(df['CalYear'])
df['IncidentGroup']=le.fit_transform(df['IncidentGroup'])
df['PropertyCategory']=le.fit_transform(df['PropertyCategory'])
df['PropertyType']=le.fit_transform(df['PropertyType'])
df['Notional Cost (£)']=le.fit_transform(df['Notional Cost (£)'])
print(df)


X1=df.iloc[:,1]
X2=df.iloc[:,2]
X3=df.iloc[:,3]
X4=df.iloc[:,4]
X5=df.iloc[:,5]
X6=df.iloc[:,6]
X7=df.iloc[:,7]
X8=df.iloc[:,8]

X=np.column_stack((X1,X2,X3,X4,X5,X6,X7,X8))
y = df.iloc[:,9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
plt.show()
model = LogisticRegression(multi_class='multinomial').fit(X_train,y_train)
#print(model.intercept_)
#print(model.coef_)
y_pred = model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
plot_confusion_matrix(model, X_test, y_test)
plt.show()