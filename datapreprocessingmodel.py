import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

# Models
logistic_regression_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()
neural_network_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Training models
logistic_regression_model.fit(X_train_tfidf, y_train)
decision_tree_model.fit(X_train_tfidf, y_train)
random_forest_model.fit(X_train_tfidf, y_train)
neural_network_model.fit(X_train_tfidf, y_train)

# Evaluating models
models = {
    "Logistic Regression": logistic_regression_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "Neural Network": neural_network_model
}

for name, model in models.items():
    print(f"Model: {name}")
    print("Training Accuracy:", model.score(X_train_tfidf, y_train))
    print("Test Accuracy:", model.score(X_test_tfidf, y_test))
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean Cross-validation score:", cv_scores.mean())
    y_pred = model.predict(X_test_tfidf)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------------------------")
