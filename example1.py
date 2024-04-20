import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from your_data.csv
try:
    data = pd.read_csv('your_data.csv')
except FileNotFoundError:
    # If the file is not found, create an empty DataFrame with the required columns
    data = pd.DataFrame(columns=['minute', 'Y', 'd', 'maximise', 'multiple_of_5', 'last_digit_3', 'last_digit_7', 'seconds_remaining'])

# Define features (X) and target variable (y)
X = data.drop('maximise', axis=1)  # Features
y = data['maximise']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, predictions))

# Example input for prediction
input_data = pd.DataFrame({
    'minute': [55],
    'Y': [1.5],
    'd': [0],
    'multiple_of_5': [False],
    'last_digit_3': [True],
    'last_digit_7': [False],
    'seconds_remaining': [30]
})

# Predict the next 2
prediction = model.predict(input_data)
print(f'Predicted next 2: {prediction}')
