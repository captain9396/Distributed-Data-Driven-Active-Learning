import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib



DATA_DIR = 'D:\DEVELOPMENT\MACHINE_LEARNING\Random Forest\car_evaluation_dataset.data'

fp = open(DATA_DIR)
raw = []

for line in fp.readlines():
    raw.append(line.strip().split(','))

raw = np.array(raw)

labelCode = {}

cols = len(raw[0,:])

val = 0
for i in range(cols):
    labels = list(set(raw[:, i]))
    for l in labels:
        if l not in labelCode:
            labelCode[l] = val
            val+=1

# print(labelCode)

data = []
for row in raw:
    nrow = []
    for attr in row:
        nrow.append(labelCode[attr])
    data.append(nrow)

data = np.array(data)




dataFrame = pd.DataFrame({
    'price' : data[: , 0],
    'maintenance' : data[: , 1],
    'doors' : data[: , 2],
    'persons' : data[: , 3],
    'lug-boot' : data[: , 4],
    'safety' : data[: , 5],
    'class' : data[: , 6],
})


featureNames = ['price', 'maintenance', 'doors', 'persons', 'lug-boot', 'safety']


X = dataFrame[featureNames]
y = dataFrame[['class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
feature_importances = pd.Series(classifier.feature_importances_, index=featureNames) .sort_values(ascending=False)


print(feature_importances)


