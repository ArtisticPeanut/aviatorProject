import pandas as pd

# Read the dataset from 'your_data.csv'
df = pd.read_csv('your_data.csv')

# Function to check conditions and return 1 if critical, 0 otherwise
def is_critical(row):
    if row['Y'] > 9.99 and str(row['minute'])[-1] in ['0','5','2','7','9']:
        return 1
    else:
        return 0

# Apply the function to create the 'is_critical' column
df['is_critical'] = df.apply(is_critical, axis=1)

# Display the updated DataFrame
print(df)

# Save the updated DataFrame back to 'your_data.csv'
df.to_csv('your_data.csv', index=False)
