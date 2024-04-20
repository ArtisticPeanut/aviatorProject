import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset from a CSV file
# Replace 'your_data.csv' with the actual file path or URL to your dataset
df = pd.read_csv('your_data.csv')

# Feature engineering: creating a binary target variable
df['target'] = (df['NextMinuteNumber'] > 1.99).astype(int)

# Select relevant features for the model
X = df[['Result1', 'Result2', 'Result3']]

# Target variable
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Now, you can use the model to make predictions for the next minute
# Let's say the user provides three results: user_result1, user_result2, user_result3
user_results = [user_result1, user_result2, user_result3]
user_prediction_input = pd.DataFrame([user_results], columns=['Result1', 'Result2', 'Result3'])

# Make a prediction for the user input
user_prediction = model.predict(user_prediction_input)

# Print the prediction
if user_prediction[0] == 1:
    print("The model predicts that the next minute's number will be greater than 1.99.")
else:
    print("The model predicts that the next minute's number will not be greater than 1.99.")
