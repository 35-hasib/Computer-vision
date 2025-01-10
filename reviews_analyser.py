import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')

# Prepare the dataset
documents = [(" ".join(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
df = pd.DataFrame(documents, columns=["reviews", "sentiment"])

# Vectorize the reviews
vectorized = CountVectorizer(max_features=2000)
X = vectorized.fit_transform(df["reviews"])
y = df["sentiment"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Define the predict_sentiment function
def predict_sentiment(reviews):
    reviews = vectorized.transform([reviews])
    return model.predict(reviews)[0]

# Test the predict_sentiment function
print(predict_sentiment("Fine movie!"))
print(predict_sentiment("Bad movie!"))
print(predict_sentiment("Great movie!"))
print(predict_sentiment("Not good"))
