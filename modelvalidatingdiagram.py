import matplotlib.pyplot as plt

# Accuracy scores
training_accuracy = logistic_model.score(X_train_tfidf, y_train)
test_accuracy = logistic_model.score(X_test_tfidf, y_test)
mean_cv_score = cv_scores.mean()

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(['Training Accuracy', 'Test Accuracy', 'Mean CV Score'], [training_accuracy, test_accuracy, mean_cv_score], color=['blue', 'green', 'orange'])
plt.title('Model Performance')
plt.xlabel('Accuracy')
plt.ylabel('Score')
plt.ylim(0, 1)  # Setting y-axis limit to better visualize accuracy scores
plt.show()
