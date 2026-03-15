import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
data = pd.read_csv(r"D:\jupyter folder\Sleep_health_and_lifestyle_dataset.csv")
data.head()
X = data.drop(columns=['Person ID', 'Occupation', 'Sleep Disorder'])
y = data['Sleep Disorder']
X = pd.get_dummies(X, columns=['Gender', 'BMI Category', 'Blood Pressure'], drop_first=True)
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]
svm_model = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
svm_model.fit(X_train_2d, y_train)
y_pred = svm_model.predict(X_test_2d)
# Compute confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Define grid for decision boundary
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict on grid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA']))
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=ListedColormap(['#FF0000', '#0000FF', '#00FF00']), edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM Decision Boundary and Data Classification")
plt.show()
