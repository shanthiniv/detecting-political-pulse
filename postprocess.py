import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Age': [25, 35, 45, 55, 65],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Party': ['A', 'B', 'A', 'B', 'A'],
    'Approval_Rating': [60, 45, 70, 55, 75]
}

df = pd.DataFrame(data)


age_distribution = df['Age'].value_counts().sort_index()
gender_distribution = df['Gender'].value_counts()

party_affiliation = df['Party'].value_counts()


average_approval = df.groupby('Party')['Approval_Rating'].mean()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
age_distribution.plot(kind='bar', color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of People')

plt.subplot(1, 2, 2)
gender_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Gender Distribution')
plt.ylabel('')

plt.tight_layout()
plt.show()


party_affiliation.plot(kind='bar', color='skyblue')
plt.title('Party Affiliation')
plt.xlabel('Party')
plt.ylabel('Number of People')
plt.show()


average_approval.plot(kind='line', marker='o', color='orange')
plt.title('Average Approval Rating by Party')
plt.xlabel('Party')
plt.ylabel('Approval Rating')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
