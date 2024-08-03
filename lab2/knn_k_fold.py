import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

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


# Initialize metrics
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
conf_matrix_sum = np.zeros((3, 3))

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(dt_properties):
    # X_train: Attribute dataset for training
    # X_test: Attribute dataset for testing
    # y_train: Label dataset for training
    # y_ test: Label dataset for testing
    X_train, X_test = dt_properties.iloc[train_index], dt_properties.iloc[test_index]
    y_train, y_test = dt_label.iloc[train_index], dt_label.iloc[test_index]

    # Build model
    KNN_Model = KNeighborsClassifier(n_neighbors=5)
    KNN_Model.fit(X_train, y_train)

    # Predict label for elements in the test set
    y_predict = KNN_Model.predict(X_test)
    # print(y_predict)

    # Calculate metrics
    accuracy_scores.append(accuracy_score(y_test, y_predict))
    f1_scores.append(f1_score(y_test, y_predict, average='macro', zero_division=0))
    precision_scores.append(precision_score(y_test, y_predict, average='macro', zero_division=0))
    recall_scores.append(recall_score(y_test, y_predict, average='macro', zero_division=0))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predict, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    conf_matrix_sum += conf_matrix

# Calculate average metrics
avg_accuracy = np.mean(accuracy_scores)
avg_f1 = np.mean(f1_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)

print("Average Accuracy score: ", avg_accuracy)
print("Average F1 score: ", avg_f1)
print("Average Precision score: ", avg_precision)
print("Average Recall score: ", avg_recall)


# Draw the chart
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(accuracy_scores) + 1), [score * 100 for score in accuracy_scores], label='Accuracy', marker='o')
plt.plot(range(1, len(precision_scores) + 1), [score * 100 for score in precision_scores], label='Precision', marker='o')
plt.plot(range(1, len(recall_scores) + 1), [score * 100 for score in recall_scores], label='Recall', marker='o')
plt.plot(range(1, len(f1_scores) + 1), [score * 100 for score in f1_scores], label='F1 Score', marker='o')

plt.title("KNN Metrics (K-Fold)")
plt.xlabel("Iteration")
plt.ylabel("Score (%)")
plt.legend()
plt.grid(True)
plt.savefig("KNN_metrics_kfold.png")
plt.show()


# Confusion matrix
print("Accumulated Confusion Matrix:\n", conf_matrix_sum)

# Draw Confusion matrix chart
class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix_sum, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Average Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')

thresh = conf_matrix_sum.max() / 2.
for i, j in itertools.product(range(conf_matrix_sum.shape[0]), range(conf_matrix_sum.shape[1])):
    plt.text(j, i, f"{conf_matrix_sum[i, j]:.2f}", horizontalalignment="center",
             color="white" if conf_matrix_sum[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("KNN_cm_k_fold.png")
plt.show()