# Create a program to analyze customer churn data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score


# 1. Data Preparation
df = pd.read_csv(r"C:\Users\acer\OneDrive\Desktop\internshif file\All_csv_file\Churn_modelling file 2.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.tail())
# print(df.info())
df

print("\n")
print(df.columns)  # Display column names


# ------------------------------------------------------------------------------------- #
# Ploting and Gender Count
import matplotlib.pyplot as plt
import seaborn as sns

# Count the number of male and female customers
df['Gender'].value_counts()
     
# Plotting the features of the dataset to see the correlation between them

plt.hist(x = df.Gender, bins = 3, color = 'pink')
plt.title('comparison of male and female')
plt.xlabel('Gender')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# Ploting and Age Count
df['Age'].value_counts()
     
# comparison of age in the dataset
plt.hist(x = df.Age, bins = 10, color = 'orange')
plt.title('comparison of Age')
plt.xlabel('Age')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# Ploting and Geography Count
df['Geography'].value_counts()

# comparison of geography in the dataset
plt.hist(x = df.Geography, bins = 3, color = 'green')
plt.title('comparison of Geography')
plt.xlabel('Geography')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# HasCard: # Ploting and HasCrCard Count
df['HasCrCard'].value_counts()

# comparision of how many customers hold the credit card
plt.hist(x = df.HasCrCard, bins = 3, color = 'red')
plt.title('how many people have or not have the credit card')
plt.xlabel('customers holding credit card')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# Ploting and IsActiveMember Count
df['IsActiveMember'].value_counts()
# comparision of how many customers are active members
plt.hist(x = df.IsActiveMember, bins = 3, color = 'blue')
plt.title('how many people are active members')
plt.xlabel('active members')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# Ploting and Exited Count
df['Exited'].value_counts()
# comparision of how many customers are exited
plt.hist(x = df.Exited, bins = 3, color = 'purple')
plt.title('how many people are exited')
plt.xlabel('exited customers')
plt.ylabel('population')
plt.show()
# ------------------------------------------------------------------------------------- #

# Step 1: Drop non-informative columns (like customerID)
X = df.drop(['CustomerId', 'Exited'], axis=1)
y = df['Exited']

X = pd.get_dummies(X)
# ------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Output : Model Accuracy: 
# ------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
sns.barplot(x=feature_importance.values[:10], y=feature_importance.index[:10])
plt.title("Top 10 Important Features")
plt.show()
# ------------------------------------------------------------------------------------- # 


# -------------------------------------------------------------------------------------- #
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# ------------------------------------------------------------------------------------- #


# Ploting Pie Chart
plt.figure(figsize=(8, 6))
plt.title('Customer Distribution by Country')

labels = 'France', 'Germany', 'Spain'
colors = ['cyan', 'magenta', 'orange']
sizes =  [311, 300, 153]
explode = [ 0.01, 0.01, 0.01]

plt.pie(sizes, colors = colors, labels = labels, explode = explode, shadow = True)

plt.axis('equal')
plt.show()
     

# -------------------------------------------------------------------------------------- #
'''
from sklearn.model_selection import GridSearchCV
# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# Fit the grid search to the training data
grid_search.fit(X_train, y_train)
# Print the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2f}")
'''
# ------------------------------------------------------------------------------------- #

'''
# 2. Data Cleaning
# Handle missing values
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df['total_charges'].fillna(df['total_charges'].median(), inplace=True)
'''
