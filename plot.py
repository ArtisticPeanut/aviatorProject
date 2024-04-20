import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("your_data.csv")

# Assuming your CSV file has columns 'minute' and 'y'
X = df[['minute']]
y = df['y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plotting the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Minute')
plt.ylabel('y')
plt.title('Linear Regression Model')
plt.show()

# Predict the next time y will be greater than 2
next_minute = 10  # Adjust this based on your requirement
predicted_y = model.predict([[next_minute]])
print(f"Predicted y at minute {next_minute}: {predicted_y[0]}")

# Alternatively, you can use the model to predict for a range of minutes
# predicted_y_values = model.predict(X_range)
# print(predicted_y_values)
