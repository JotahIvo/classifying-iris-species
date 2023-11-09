from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print(iris_dataset['data'])
#print(iris_dataset['target'])


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


import pandas as pd 
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#creating a scatter matrix from the dataframe
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)


#k-neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#making predictions
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]]) # it is an example of a wild iris (just an example)

prediction = knn.predict(X_new)
print(f"Prediction: {prediction}")
print(f"Prediction target name: {iris_dataset['target_names'][prediction]}")


#evaluating the model (comparing the test dataset with the correct answers)
y_pred= knn.predict(X_test)
print("\nEvaluating the model:")
print(f"Test set predictions: {y_pred}")
print(f"Test set score: {np.mean(y_pred == y_test)}")

#we can use the score method os knn object: knn.score(X_test, y_test)
