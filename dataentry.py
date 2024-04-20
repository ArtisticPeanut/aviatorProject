import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import keyboard

# Load the trained model
try:
    user_inputs = pd.read_csv('your_data.csv')
except FileNotFoundError:
    user_inputs = pd.DataFrame(columns=['minute', 'Y', 'd', 'maximise', 'multiple_of_5', 'last_digit_3', 'last_digit_7', 'seconds_remaining'])

while True:
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
    maximise = 1 if user_inputs.empty or (not user_inputs.empty and user_inputs['Y'].iloc[-1] < 2) else 0

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

    user_inputs = pd.concat([user_inputs, new_data], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    user_inputs.to_csv('your_data.csv', index=False)

    print("Data added successfully!")

    # Check if the user wants to continue
    if keyboard.is_pressed('Esc'):
        break

# Predict using the model
# Add your prediction code here based on the trained model
