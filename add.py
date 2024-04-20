import pandas as pd

# Sample data for illustration
data = pd.read_csv('your_data.csv')

df = pd.DataFrame(data)

# Function to check conditions and return 1 if critical, 0 otherwise
def is_critical(row):
    if row['Y'] > 9.99 and str(row['minute'])[-1] in ['0', '5', '2', '7']:
        return 1
    else:
        return 0

# Apply the function to create the 'is_critical' column
df['is_critical'] = df.apply(is_critical, axis=1)

# Display the updated DataFrame
print(df)
