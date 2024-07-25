import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

dataset = pd.read_csv("./iris_data.csv", delimiter=",")

# Print properties name of csv file
# print(dataset.columns)

# Print the first 5 rows
# print(dataset.head(5))
# print(dataset.loc[0:4])

# Print all rows of column 3 & 4
# print(dataset.iloc[:, 3:5])

# Print all columns of row 3 and 4
# print(dataset.iloc[3:5, :])

# print(dataset.iloc[0:2, 3:5])

# Split the dataset into 2 parts Properties (thuộc tính) - Label (nhãn)
dt_properties = dataset.iloc[:, 0: 4]
dt_label = dataset.species

# X_train: Attribute dataset for training
# X_test: Attribute dataset for testing
# y_train: Label dataset for training
# y_ test: Label dataset for testing
X_train, X_test, y_train, y_test = train_test_split(dt_properties, dt_label, test_size=1 / 3, random_state=42)

# Build model
KNN_Model = KNeighborsClassifier(n_neighbors=5)
KNN_Model.fit(X_train, y_train)

# Predict label for elements in the test set
y_predict = KNN_Model.predict(X_test)
# print(y_predict)

# Calculate Accuracy, Confusion Matrix
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')

print("Accuracy score: ", accuracy)
print("F1 score", f1)
print("Precision score", precision)
print("Recall score", recall)

# Draw the chart
plt.figure(figsize=(8, 6))
plt.plot([1], [accuracy * 100], label='Accuracy', marker='o')
plt.plot([1], [precision * 100], label='Precision', marker='o')
plt.plot([1], [recall * 100], label='Recall', marker='o')
plt.plot([1], [f1 * 100], label='F1 Score', marker='o')

plt.title("KNN Metrics")
plt.xlabel("Iteration")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.savefig("KNN_metrics.png")
plt.show()


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print("Confusion matrix: ", conf_matrix)

# Draw Confusion matrix chart
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')

thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, f"{conf_matrix[i, j]:.2f}", horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("KNN_confusion_matrix.png")
plt.show()