from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
import datetime
from datetime import timedelta

t1 = datetime.datetime.now()





DATA_DIRECTORY = 'D:/DEVELOPMENT/MACHINE_LEARNING/creditcardfraud/'
fp = open(DATA_DIRECTORY + 'creditcard.csv')

X = []
y = []

data = fp.readlines()
for l in data[1:]:
    line = l.strip().split(',')
    X.append(list(map(float, line[:-1])))
    y.append(float(line[-1][1]))


X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)



model = RandomForestClassifier(n_estimators=100)
print('Building model ...')
model.fit(X_train, y_train)

# saves model
joblib.dump(model, 'D:/DEVELOPMENT/MACHINE_LEARNING/creditcardfraud/credit_card_fraud_model.joblib')

# loads model from persistence storage
# model = joblib.load('D:/DEVELOPMENT/MACHINE_LEARNING/creditcardfraud/credit_card_fraud_model.joblib')



pred = model.predict(X_test)

score = metrics.accuracy_score(y_test, pred)

print('Accuracy score = ' , score)





# calculate time needed
t2 = datetime.datetime.now()
time_difference = t2 - t1
time_difference_in_minutes = time_difference / timedelta(minutes=1)
print('Time elapsed = ' , time_difference_in_minutes , ' minutes')