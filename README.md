# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step 1:
Import libraries – Import pandas, matplotlib, and sklearn modules.
## Step 2:
Load dataset – Read the dataset containing Position Level and Salary.
## Step 3:
Split data – Divide the data into training and testing sets.
## Step 4:
Train model – Create and train a Decision Tree Regressor on the training data.
## Step 5:
Predict & evaluate – Predict salaries for test data, display actual vs predicted results, and visualize the Decision Tree.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Kanigavel M 
RegisterNumber: 212224240070  
*/

# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# The dataset should have columns: Level, Salary
data = pd.read_csv('salary.csv')
print("Dataset Head:\n", data.head())

# Step 2: Define features and target
X = data[['Level']]   # Independent variable (Position Level)
y = data['Salary']    # Dependent variable (Salary)

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Create and train the Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=0, max_depth=4)
regressor.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = regressor.predict(X_test)

# Step 6: Display results in a table
results = pd.DataFrame({
    'Position Level': X_test['Level'],
    'Actual Salary': y_test,
    'Predicted Salary': y_pred
}).sort_values(by='Position Level')

print("\nPrediction Results:")
print(results.to_string(index=False))

# Step 7: Evaluate model performance
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(15,10))
plot_tree(regressor, filled=True, feature_names=['Level'], rounded=True, fontsize=10)
plt.title('Decision Tree for Employee Salary Prediction (Based on Position Level)')
plt.show()

```

## Output:
<img width="655" height="377" alt="image" src="https://github.com/user-attachments/assets/f757393a-4187-4d5c-829b-017c88fe5ddd" />

<img width="1394" height="846" alt="image" src="https://github.com/user-attachments/assets/69e0f961-e686-466c-9c55-591699484208" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
