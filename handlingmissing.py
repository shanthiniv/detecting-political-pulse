import pandas as pd

data = {
    'ID': [1, 2, 3, 4, 5],
    'Age': [25, 35, None, 55, 65],  
    'Gender': ['Male', 'Female', 'Male', None, 'Male'], 
    'Party': ['A', 'B', 'A', 'B', 'A'],
    'Approval_Rating': [60, None, 70, 55, 75]  
}

df = pd.DataFrame(data)

df.dropna(inplace=True)

print("Preprocessed Data:")
print(df)
