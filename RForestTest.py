from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics


iris = datasets.load_iris()




data = pd.DataFrame({
    'sepal length' : iris.data[:,0],
    'sepal width' : iris.data[:,1],
    'petal length' : iris.data[:,2],
    'petal width' : iris.data[:,3],
    'species' : iris.target
})


X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data[['species']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



classifier = RandomForestClassifier(n_estimators=100)

classifier.fit(X_train, y_train)

y_pred= classifier.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))

feature_importances = pd.Series(classifier.feature_importances_, index=iris.feature_names) .sort_values(ascending=False)

print(feature_importances)

