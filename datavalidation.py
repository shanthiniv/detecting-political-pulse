# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load datasets
bjp_data = pd.read_csv("bjp_tweets.csv")
congress_data = pd.read_csv("congress_tweets.csv")

# Concatenate datasets and create labels
bjp_data['target'] = 'bjp'
congress_data['target'] = 'congress'
combined_data = pd.concat([bjp_data, congress_data], ignore_index=True)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_data['tweet'], combined_data['target'], test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training a logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Evaluating the model
print("Training Accuracy:", logistic_model.score(X_train_tfidf, y_train))
print("Test Accuracy:", logistic_model.score(X_test_tfidf, y_test))

# Cross-validation
cv_scores = cross_val_score(logistic_model, X_train_tfidf, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean Cross-validation score:", cv_scores.mean())

# Classification report
y_pred = logistic_model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
