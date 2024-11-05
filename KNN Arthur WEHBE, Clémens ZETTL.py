# ce code est une preuve de rendu à l'heure de notre travail, n'étant pas sur que mon camarade ai pensé à l'inclure dans le rapport

import numpy as np
import joblib
import pandas as pd

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_size = int(test_size * len(X))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

class KNeighborsClassifierCustom:
    def __init__(self, n_neighbors=100):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_neighbors_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_neighbors_indices]
            predicted_label = np.bincount(nearest_labels).argmax()
            predictions.append(predicted_label)

        return np.array(predictions)

def accuracy_score_custom(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# on charge les données depuis le fichier dataset :
data = pd.read_excel('C:\Devoir Arthur\DataScience et IA\dataset.xlsx', engine='openpyxl')

# on divise les données de test et d'entraînement :
X = np.array(data.iloc[:, :-1])
y = np.array(data.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

# classifier avec k = 25 :

knn_classifier = KNeighborsClassifierCustom(n_neighbors=25)

# on entraine :
    
knn_classifier.fit(X_train, y_train)

# on fait la prédiciton :
y_pred = knn_classifier.predict(X_test)

# précision du modèle :
accuracy = accuracy_score_custom(y_test, y_pred)
print(f"Précision du modèle : {accuracy}")

joblib.dump(knn_classifier, "modele_knn.pkl")
knn_model = joblib.load("modele_knn.pkl")
