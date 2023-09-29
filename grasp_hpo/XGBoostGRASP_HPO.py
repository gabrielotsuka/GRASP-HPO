import uuid

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from queue import PriorityQueue
import random


def prepare_dataset(dataset):
    x = dataset.data
    y = dataset.target
    return train_test_split(x, y, test_size=0.2, random_state=1)


x_train, x_test, y_train, y_test = prepare_dataset(load_breast_cancer())


def evaluate_solution(params):
    xgboost_classifier = XGBClassifier(**params)
    xgboost_classifier.fit(x_train, y_train)
    y_pred = xgboost_classifier.predict(x_test)
    return f1_score(y_test, y_pred, average='weighted')


hyperparameter_ranges = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}


def building_phase():
    global x_train, x_test
    number_of_iterations = 20
    best_intermediate_combinations = PriorityQueue()
    intermediate_results_size = 3
    for i in range(0, number_of_iterations):

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        selected_hyperparameters = {
            'n_estimators': random.randint(
                    hyperparameter_ranges['n_estimators'][0],
                    hyperparameter_ranges['n_estimators'][1]
            ),
            'max_depth': random.randint(
                    hyperparameter_ranges['max_depth'][0],
                    hyperparameter_ranges['max_depth'][1]
            ),
            'colsample_bytree': random.uniform(
                hyperparameter_ranges['colsample_bytree'][0],
                hyperparameter_ranges['colsample_bytree'][1]
            ),
            'reg_lambda': random.uniform(
                hyperparameter_ranges['reg_lambda'][0],
                hyperparameter_ranges['reg_lambda'][1]
            ),
            'subsample': random.uniform(
                hyperparameter_ranges['subsample'][0],
                hyperparameter_ranges['subsample'][1]
            )
        }

        print(selected_hyperparameters)

        f1_score = evaluate_solution(selected_hyperparameters)

        best_intermediate_combinations.put((f1_score, uuid.uuid4(), selected_hyperparameters))
        if best_intermediate_combinations.qsize() > intermediate_results_size:
            best_intermediate_combinations.get()
        print(f1_score)

    return best_intermediate_combinations


best_intermediate_combinations = building_phase()


def hill_climb(current_solution):
    max_iterations = 100
    best_solution = current_solution
    best_score = evaluate_solution(current_solution)

    for i in range(max_iterations):
        print(i)
        neighbor_solution = current_solution.copy()
        param_to_perturb = random.choice(list(neighbor_solution.keys()))

        if param_to_perturb in ['n_estimators', 'max_depth']:
            neighbor_solution[param_to_perturb] = random.randint(*hyperparameter_ranges[param_to_perturb])
        else:
            neighbor_solution[param_to_perturb] = random.uniform(*hyperparameter_ranges[param_to_perturb])

        neighbor_score = evaluate_solution(neighbor_solution)

        if neighbor_score > best_score:
            best_solution = neighbor_solution
            best_score = neighbor_score

        current_solution = neighbor_solution

    return best_score, best_solution


outer_counter = 1
print(str(outer_counter) + "---------")
print()
local_best_score, local_best_solution = hill_climb(best_intermediate_combinations.get()[2])
while not best_intermediate_combinations.empty():
    outer_counter = outer_counter + 1
    print(str(outer_counter) + "---------")
    print()
    temporary_score, temporary_solution = hill_climb(best_intermediate_combinations.get()[2])
    if local_best_score < temporary_score:
        local_best_score = temporary_score
        local_best_solution = temporary_solution


print("Hyperparameters: " + str(local_best_solution))
print("Achieved best score: " + str(local_best_score))

