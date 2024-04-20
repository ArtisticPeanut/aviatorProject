import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Load the Data
try:
    your_data = pd.read_csv('your_data.csv')
except FileNotFoundError:
    print("CSV file not found. Please provide the correct path or create the file.")

# Data Preparation
X = your_data[['minute', 'd', 'maximise', 'multiple_of_5', 'last_digit_3', 'last_digit_7', 'seconds_remaining']]
y = your_data['Y']

# Train the Model
model = RandomForestRegressor()
model.fit(X, y)

# Model Evaluation (optional, if you have a test set)
predictions = model.predict(X)
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict with New Data
past_Y_values = [float(input(f"Enter past Y value {i + 1}: ")) for i in range(3)]
time_laps = float(input("Enter time laps: "))
new_d = int(input("Enter 'd' value (0 or 1): "))
new_maximise = int(input("Enter 'maximise' value (0 or 1): "))
new_minute_time = int(input("Enter the minute time: "))

# Feature Engineering
new_multiple_of_5 = new_minute_time % 5 == 0
new_last_digit_3 = new_minute_time % 10 == 3
new_last_digit_7 = new_minute_time % 10 == 7
new_seconds_remaining = 60 - time_laps

# Create new DataFrame
new_data = pd.DataFrame({
    'minute': [new_minute_time],
    'd': [new_d],
    'maximise': [new_maximise],
    'multiple_of_5': [new_multiple_of_5],
    'last_digit_3': [new_last_digit_3],
    'last_digit_7': [new_last_digit_7],
    'seconds_remaining': [new_seconds_remaining]
})

# Predict the next Y
predicted_Y = model.predict(new_data)
print(f'Predicted Y: {predicted_Y}')
