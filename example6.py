import pandas as pd
from sklearn.model_selection import train_test_split
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

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
current_time = time.localtime()
minute = current_time.tm_min
current_seconds = current_time.tm_sec
# Predictions based on the next 3 consecutive instances
consecutive_results = []

for _ in range(3):
    new_minute = current_time.tm_min
    new_d = int(input("Enter 'd' value (0 or 1): "))
    new_maximise = int(input("Enter 'maximise' value (0 or 1): "))
    new_multiple_of_5 = new_minute % 5 == 0
    new_last_digit_3 = new_minute % 10 == 3
    new_last_digit_7 = new_minute % 10 == 7
    new_seconds_remaining = 60 - current_time.tm_sec

    new_data = pd.DataFrame({
        'minute': [new_minute],
        'd': [new_d],
        'maximise': [new_maximise],
        'multiple_of_5': [new_multiple_of_5],
        'last_digit_3': [new_last_digit_3],
        'last_digit_7': [new_last_digit_7],
        'seconds_remaining': [new_seconds_remaining]
    })

    predicted_Y = model.predict(new_data)
    print(f'Predicted Y: {predicted_Y[0]}')

    # Append the new data and prediction to the dataset
    new_data['Y'] = predicted_Y[0]
    your_data = pd.concat([your_data, new_data], ignore_index=True)
    consecutive_results.append(predicted_Y[0])

# Print the consecutive results
print(f'Consecutive Results: {consecutive_results}')

# Save the updated DataFrame to the CSV file
your_data.to_csv('your_data.csv', index=False)
