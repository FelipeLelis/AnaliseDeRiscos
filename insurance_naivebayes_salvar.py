import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

base = pd.read_csv('insurance.csv')
base.Age.unique()
base.RiskAversion.unique()
base.MakeModel.unique()
base.Accident.unique()

X = base.iloc[:, [2, 4, 9]].values
y = base.iloc[:, 8].values

labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])

modelo = GaussianNB()
modelo.fit(X, y)

previsoes = modelo.predict(X)
accuracy_score(y, previsoes)

pickle.dump(modelo, open('naivebayes_finalizado.sav', 'wb'))