import uuid

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from queue import PriorityQueue
import random

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

hyperparameters = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}
print(hyperparameters)

numberOfIterations = 20
print()

bestIntermediateResults = PriorityQueue()
intermediateResultsSize = 3

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

for i in range(0, numberOfIterations):

    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    selected_hyperparameters = {
        'n_estimators': random.uniform(hyperparameters['n_estimators'][0], hyperparameters['n_estimators'][1]),
        'max_depth': random.uniform(hyperparameters['max_depth'][0], hyperparameters['max_depth'][1]),
        'colsample_bytree': random.uniform(hyperparameters['colsample_bytree'][0], hyperparameters['colsample_bytree'][1]),
        'reg_lambda': random.uniform(hyperparameters['reg_lambda'][0], hyperparameters['reg_lambda'][1]),
        'subsample': random.uniform(hyperparameters['subsample'][0], hyperparameters['subsample'][1])
    }

    print(selected_hyperparameters)

    xgboostClassifier = XGBClassifier(
        n_estimators=int(selected_hyperparameters['n_estimators']),
        max_depth=int(selected_hyperparameters['max_depth']),
        colsample_bytree=selected_hyperparameters['colsample_bytree'],
        reg_lambda=selected_hyperparameters['reg_lambda'],
        subsample=selected_hyperparameters['subsample']
    )

    xgboostClassifier.fit(xTrain, yTrain)
    yPred = xgboostClassifier.predict(xTest)
    f1Score = f1_score(yTest, yPred, average='weighted')

    bestIntermediateResults.put((f1Score, uuid.uuid4(), selected_hyperparameters))
    if bestIntermediateResults.qsize() > intermediateResultsSize:
        bestIntermediateResults.get()
    print(f1Score)

print("\n\n##############################")
while bestIntermediateResults.not_empty:
    print(bestIntermediateResults.get())

