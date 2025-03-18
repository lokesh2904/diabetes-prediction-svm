import pandas as pd
import numpy as np
import joblib  # Import joblib for saving the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("diabetes.csv")

# Split features and target variable
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
}

# Perform Grid Search with Cross-Validation
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model and hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model and scaler
joblib.dump(best_model, "svm_model_diabetes.pkl")  # Save trained SVM model
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Print results
print("Best Hyperparameters:", best_params)
print("Test Accuracy after Hyperparameter Tuning:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model and Scaler saved successfully!")
