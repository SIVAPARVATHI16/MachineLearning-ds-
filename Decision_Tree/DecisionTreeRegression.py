import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate Synthetic Dataset
np.random.seed(42)

X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Visualize Dataset
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Data')
plt.title("Synthetic Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.savefig('dataset.png')
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create and Train Decision Tree Regressor
regressor = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("-" * 30)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Generate Smooth Curve for Visualization
X_grid = np.arange(
    X.min(),
    X.max(),
    0.01
).reshape(-1, 1)

y_grid_pred = regressor.predict(X_grid)

# Plot Predictions
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Actual Data')
plt.plot(X_grid, y_grid_pred, linewidth=2, label='Decision Tree Prediction')
plt.title("Decision Tree Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.savefig('Decision Tree Regression.png')
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    regressor,
    feature_names=["Feature"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Regressor Structure")
plt.savefig('DecisionTreeStructure.png')
plt.show()

# Print Tree Rules
print("\nDecision Tree Rules:")
print("-" * 50)

tree_rules = export_text(
    regressor,
    feature_names=["Feature"]
)

print(tree_rules)