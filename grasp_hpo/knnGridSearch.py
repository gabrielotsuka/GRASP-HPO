from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

paramGrid = {'n_neighbors': np.arange(1, 30)}
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, paramGrid, cv=5)
grid_search.fit(xTrain, yTrain)

# Best Parameters
best_k = grid_search.best_params_['n_neighbors']
print("Best k value:", best_k)

knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(xTrain, yTrain)
y_pred = knn_classifier.predict(xTest)

accuracy = accuracy_score(yTest, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(yTest, y_pred, target_names=iris.target_names)
print("Classification Report:\n", classification_rep)
