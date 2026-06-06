import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
data = pd.read_csv('BMI.csv')

# View first rows
print("Dataset Preview:")
print(data.head())

# Convert categorical variable (Gender) into numeric
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select independent variables
X = data[['Gender', 'Height', 'Weight']]

# Ensure no missing values (important for VIF)
X = X.dropna()

# Create VIF DataFrame
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Calculate VIF for each feature
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

# Display results
print("\nVariance Inflation Factor (VIF):")
print(vif_data)