import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the datasets
bjp_df = pd.read_csv('bjp_tweets.csv')  # Assuming the file name is 'bjp_dataset.csv'
congress_df = pd.read_csv('congress_tweets.csv')  # Assuming the file name is 'congress_dataset.csv'

# Combine the datasets
combined_df = pd.concat([bjp_df, congress_df], ignore_index=True)

# Data Cleaning (if required)
# For example, removing special characters, URLs, etc.

# Split data into features and target
X = combined_df['tweet']
y = combined_df['target']

# Text Vectorization using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets (optional)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
