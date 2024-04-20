import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time as t

# Assuming df is your DataFrame containing the dataset
df = pd.read_csv('your_data.csv')

# Replace 'target_column' with the actual name of your target column
target_column = 'Y'

# Define features and target variable
features = ['minute', 'd', 'multiple_of_5', 'seconds_remaining']

# Extract features and target variable
X = df[features]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Get current time information
current_time = t.localtime()
current_minute = current_time.tm_min
current_seconds = current_time.tm_sec

# Calculate seconds remaining in the current minute
seconds_remaining = 60 - current_seconds

# Collect user input for the previous 'Y' value and new 'Y' value
previous_y = float(input("Enter the previous Y value: "))
new_d = int(input("Enter d value: "))
new_5 = current_minute % 5 == 0

# Create a new data point with the features and the new minute value
new_data = pd.DataFrame([[current_minute, new_d, new_5, seconds_remaining]], columns=features)

# Make predictions for the new data point
new_prediction = model.predict(new_data)

print(f'Predicted Y for new minute value {current_minute}: {new_prediction[0]:.2f}')

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
