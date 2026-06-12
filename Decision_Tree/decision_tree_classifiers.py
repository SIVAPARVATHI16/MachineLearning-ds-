from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=99,
    stratify=y
)

# Baseline Model
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Baseline Accuracy:",
      accuracy_score(y_test, y_pred))

# Hyperparameter Grid
param_grid = {
    'max_depth': range(1, 10),
    'min_samples_leaf': range(1, 20, 2),
    'min_samples_split': range(2, 20, 2),
    'criterion': ['gini', 'entropy']
}

# Grid Search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=1),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)

# Test Set Evaluation
y_pred_best = best_model.predict(X_test)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred_best))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best,
                            target_names=data.target_names))

# Visualize Best Tree
plt.figure(figsize=(18, 12))
plot_tree(
    best_model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True
)

plt.title("Optimized Decision Tree")
plt.savefig('Optimized decision tree.png')
plt.show()