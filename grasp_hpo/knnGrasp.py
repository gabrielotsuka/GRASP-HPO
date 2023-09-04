from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from queue import PriorityQueue

iris = load_iris()
x = iris.data
y = iris.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

hyperparameters = {'n_neighbors': np.arange(1, 30, 5)}
intermediateResultsSize = 3
print('Full array: ' + str(hyperparameters['n_neighbors']))
print()

for hyperparameter in hyperparameters:
    # building phase
    bestIntermediateResults = PriorityQueue()
    windowMinValue = 0
    windowMaxValue = hyperparameters[hyperparameter].size
    print('Exploratory accuracy: ')
    for idx, hyperparameterValue in enumerate(hyperparameters[hyperparameter]):
        knn_classifier = KNeighborsClassifier(n_neighbors=hyperparameterValue)
        knn_classifier.fit(xTrain, yTrain)
        yPred = knn_classifier.predict(xTest)
        f1Socre = f1_score(yTest, yPred, average='weighted')

        print(str(hyperparameterValue) + ' -> ' +
              str(hyperparameters[hyperparameter][max(windowMinValue, idx-1):min(idx+2, windowMaxValue)]) + ': ' +
              str(f1Socre))

        bestIntermediateResults.put((f1Socre, hyperparameterValue))
        if bestIntermediateResults.qsize() > intermediateResultsSize:
            bestIntermediateResults.get()

    print()
    print('Best windows:')
    # local search phase
    best = (-1, 0)
    while not bestIntermediateResults.empty():
        window = bestIntermediateResults.get()
        print(window)
        for i in range(int(window[0] + 1.0), int(window[1])):
            knn_classifier = KNeighborsClassifier(n_neighbors=i)
            knn_classifier.fit(xTrain, yTrain)
            yPred = knn_classifier.predict(xTest)
            f1Socre = f1_score(yTest, yPred, average='weighted')
            if f1Socre > best[0]:
                best = (f1Socre, i)

    print('Melhor hyperparameter: ' + str(best[1]))
