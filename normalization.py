import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'ID': [1, 2, 3, 4, 5],
    'Age': [25, 35, 45, 55, 65],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Party': ['A', 'B', 'A', 'B', 'A'],
    'Approval_Rating': [60, 45, 70, 55, 75]
}

df = pd.DataFrame(data)
df.drop('ID', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Gender', 'Party'])
scaler = MinMaxScaler()
df[['Age', 'Approval_Rating']] = scaler.fit_transform(df[['Age', 'Approval_Rating']])

print("Preprocessed Data:")
print(df)
