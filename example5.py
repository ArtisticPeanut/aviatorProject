import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import keyboard

# Load the trained model
try:
    your_data = pd.read_csv('your_data.csv')
except FileNotFoundError:
    your_data = pd.DataFrame(columns=['minute', 'Y', 'd', 'maximise', 'multiple_of_5', 'last_digit_3', 'last_digit_7', 'seconds_remaining'])

while True:
    # Measure the time lapse since the previous entry
    start_time = time.time()

    # Calculate current minute and second
    current_time = time.localtime()
    minute = current_time.tm_min
    current_seconds = current_time.tm_sec

    # Feature engineering
    multiple_of_5 = minute % 5 == 0
    last_digit_3 = minute % 10 == 3
    last_digit_7 = minute % 10 == 7

    # Calculate seconds remaining
    seconds_remaining = 60 - current_seconds

    # Automatically set 'd' value based on seconds_remaining
    d = 1 if seconds_remaining < 10 else 0

    # Take user input for other features
    Y = float(input("Enter multiplier result (Y): "))
    
    # Automatically set 'maximise' based on the previous 'Y'
    maximise = 1 if your_data.empty or (not your_data.empty and your_data['Y'].iloc[-1] < 2) else 0

    # Append user input to the DataFrame
    new_data = pd.DataFrame({
        'minute': [minute],
        'Y': [Y],
        'd': [d],
        'maximise': [maximise],
        'multiple_of_5': [multiple_of_5],
        'last_digit_3': [last_digit_3],
        'last_digit_7': [last_digit_7],
        'seconds_remaining': [seconds_remaining]
    })

    your_data = pd.concat([your_data, new_data], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    your_data.to_csv('your_data.csv', index=False)

    print("Data added successfully!")

    # Calculate and print the time lapse
    end_time = time.time()
    time_lapse = end_time - start_time
    print(f"Time lapse since previous entry: {time_lapse} seconds")

    # Check if the user wants to continue
    if keyboard.is_pressed('Esc'):
        break

# Predict using the model
# Add your prediction code here based on the trained model
