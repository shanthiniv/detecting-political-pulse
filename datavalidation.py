import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
n_samples = 1000
economic_growth = np.random.normal(3, 1, n_samples)  # Mean 3, Standard deviation 1
approval_rating = np.random.normal(50, 10, n_samples)  # Mean 50, Standard deviation 10
election_outcome = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])  # 0: Lost, 1: Won

data = pd.DataFrame({
    'Economic Growth': economic_growth,
    'Approval Rating': approval_rating,
    'Election Outcome': election_outcome
})

X = data[['Economic Growth', 'Approval Rating']]
y = data['Election Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
