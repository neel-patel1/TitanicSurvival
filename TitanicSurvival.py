import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Load data from file
file_path = "C:/Users/Neel/Desktop/2023-2024/Spring 2024/Big Data and Forecasting/assignment3.csv"
data1 = pd.read_csv(file_path)
#print(data1)

#Data cleaning process
data1 = data1.dropna()
data1 = data1.drop(["Ticket", "PassengerId", "Cabin", "Name", "Embarked"], axis = 1)
#print(data1)


#Make sex variable binary
data1["Sex"] = data1["Sex"].map({"male": 0, "female": 1})
#print(data1)

#Make Survived variable the dependent variable
x = sm.add_constant(data1.drop(["Survived"], axis=1))
y = data1["Survived"]
#print(y)
#print(x)

# CART
tree = DecisionTreeClassifier(random_state = 42)
accuracy = cross_val_score(tree, x, y, cv=10)
print("Accuracy Score" , accuracy.mean()) 
tree.fit(x,y)

# Extract feature importances
feature_importances_tree = tree.feature_importances_

# Display feature importances as a table
importance_df_tree = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances_tree})
importance_df_tree = importance_df_tree.sort_values(by='Importance', ascending=False)
print(importance_df_tree)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df_tree['Feature'], importance_df_tree['Importance'], color='green')
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.show()

# Probit Model (using statsmodels)
probit_model = sm.Probit(y, sm.add_constant(x))
probit_results = probit_model.fit()
accuracy_probit = (probit_results.predict(x) >= 0.5).astype(int) == y
print("Probit Model Accuracy Score:", accuracy_probit.mean())

# Extract coefficients (feature importance for logistic regression)
feature_importances_probit = probit_results.params[1:]  # exclude the constant term

# Display feature importances as a table
importance_df_probit = pd.DataFrame({'Feature': x.columns[1:], 'Coefficient': feature_importances_probit})
importance_df_probit = importance_df_probit.sort_values(by='Coefficient', ascending=False)
print(importance_df_probit)

# Pruned Decision Tree

# Perform cross-validated search for the best alpha
alpha_values = np.arange(0, .5, .01)
#Test values above from 0 to 20, and reduced it to .5 to make the process quicker (knowing the value is .02)
mean_accuracy_scores = []
for alpha in alpha_values:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    accuracy_scores = cross_val_score(tree, x, y, cv=10)
    mean_accuracy = np.mean(accuracy_scores)
    mean_accuracy_scores.append(mean_accuracy)

# Find the best alpha based on mean accuracy scores
best_alpha_index = np.argmax(mean_accuracy_scores)
best_alpha = alpha_values[best_alpha_index]
print(f"Best alpha value for Pruned Model: {best_alpha}")

pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_tree.fit(x, y)
print("Pruned Model Accuracy Score:", mean_accuracy_scores[np.argmax(mean_accuracy_scores)])

# Extract feature importances
feature_importances_pruned_tree = pruned_tree.feature_importances_

# Display feature importances as a table
importance_df_pruned_tree = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances_pruned_tree})
importance_df_pruned_tree = importance_df_pruned_tree.sort_values(by='Importance', ascending=False)
print(importance_df_pruned_tree)

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
accuracy_rf = cross_val_score(random_forest, x, y, cv=10)
print("Random Forest Accuracy Score:", accuracy_rf.mean())
random_forest.fit(x,y)
# Extract feature importances
feature_importances = random_forest.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.show()

# Display feature importances as a table
print(importance_df)

# Compare performance
model_performance = {
    "Decision Tree": accuracy.mean(),
    "Probit Model": accuracy_probit.mean(),
    "Pruned Decision Tree": mean_accuracy_scores[np.argmax(mean_accuracy_scores)],
    "Random Forest": accuracy_rf.mean()
}

best_model = max(model_performance, key=model_performance.get)
print(f"\nThe best-performing model is: {best_model} with an accuracy of {model_performance[best_model]}")
