from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from queue import PriorityQueue
import random

iris = load_iris()
x = iris.data
y = iris.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

hyperparameters = {
    'n_neighbors': np.arange(1, 30),
    'weights': {'uniform', 'distance'},
    'algorithm': {'auto', 'ball_tree', 'kd_tree', 'brute'},
    'leaf_size': np.arange(1, 30)
}
print(hyperparameters)

numberOfIterations = 20
intermediateResultsSize = 3
print()

bestIntermediateResults = PriorityQueue()

for i in range(0, numberOfIterations):
    selected_hyperparameters = {
        'n_neighbors': random.choice(hyperparameters['n_neighbors']),
        'weights': random.choice(list(hyperparameters['weights'])),
        'algorithm': random.choice(list(hyperparameters['algorithm'])),
        'leaf_size': random.choice(hyperparameters['leaf_size'])
    }

    print(selected_hyperparameters)

    knn_classifier = KNeighborsClassifier(
        n_neighbors=selected_hyperparameters['n_neighbors'],
        weights=selected_hyperparameters['weights'],
        algorithm=selected_hyperparameters['algorithm'],
        leaf_size=selected_hyperparameters['leaf_size']
    )

    knn_classifier.fit(xTrain, yTrain)
    yPred = knn_classifier.predict(xTest)
    f1Score = f1_score(yTest, yPred, average='weighted')

    print(f1Score)

    bestIntermediateResults.put((f1Score, selected_hyperparameters))
    if bestIntermediateResults.qsize() > intermediateResultsSize:
        bestIntermediateResults.get()

print(bestIntermediateResults)
