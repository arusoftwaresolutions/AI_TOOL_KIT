"""
Iris Classification with Decision Tree
- Uses sklearn's iris dataset
- Introduces artificial missing values, imputes them
- Encodes labels, trains DecisionTreeClassifier
- Evaluates accuracy, precision, recall and plots confusion matrix
Run:
    python task1_iris_sklearn.py
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Reproducible
np.random.seed(42)

# Load dataset
iris = datasets.load_iris()
X = iris.data.copy()
y = iris.target.copy()
target_names = iris.target_names

# Introduce a few artificial missing values for demonstration
mask = np.random.rand(*X.shape) < 0.05
X[mask] = np.nan

# Impute missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Encode labels (already numeric but we show encoder usage)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

print("Iris Decision Tree Results")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Iris Decision Tree')
plt.tight_layout()
plt.show()

# Optional: visualize tree
plt.figure(figsize=(12,6))
plot_tree(clf, feature_names=iris.feature_names, class_names=target_names, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.show()