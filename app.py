# Install necessary packages
# !pip install pandas matplotlib seaborn wordcloud nltk contractions emoji beautifulsoup4 scikit-learn

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import re
import contractions
import emoji
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('semental/Sentiment-Analysis-of-Restaurant-Reviews/Reviews.csv')
print(data.head(3))
print(data.tail(1))
print(data.info())
print(data.describe())

# Check for null values
print(data.isnull().sum())
print(data.duplicated().sum())

# Value counts of the target variable
value_counts = data['Liked'].value_counts()
print(value_counts)

# Plot value counts
value_counts.plot(kind='bar', color=['blue', 'green'])
plt.title("Sentiment Value Counts")
plt.xlabel('Liked')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Positive', 'Negative'], rotation=0)
plt.show()

# Word cloud of reviews
combined_text = " ".join(data['Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

# Frequency of specific words in reviews
target_words = ['food', 'place', 'restaurant']
all_words = " ".join(data['Review']).lower().split()
word_counts = Counter(all_words)
target_word_counts = {word: word_counts[word] for word in target_words}
plt.figure(figsize=(8, 6))
plt.bar(target_word_counts.keys(), target_word_counts.values(), color=['blue', 'green', 'orange'])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Frequency of Specific Words in Reviews')
plt.show()

# Text preprocessing
# Convert to lowercase
data['Review'] = data['Review'].str.lower()
print(data['Review'])

# Tokenization
data['Tokens'] = data['Review'].apply(word_tokenize)
print(data['Tokens'])
print(data.info())

# Remove punctuation
import string
data['Review'] = data['Review'].str.replace(f"[{string.punctuation}]", " ", regex=True)
print(data['Review'])

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['Tokens'] = data['Review'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])
print(data['Tokens'])

# Stemming
stemmer = PorterStemmer()
data['Stemmed'] = data['Review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))
print(data['Stemmed'])

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['Lemmatized'] = data['Review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in word_tokenize(x)]))
print(data['Lemmatized'])

# Remove numbers
data['No_Numbers'] = data['Review'].apply(lambda x: re.sub(r'\d+', ' ', x))
print(data['No_Numbers'])

# Remove special characters
data['Cleaned'] = data['Review'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))
print(data['Cleaned'])

# Expand contractions
data['Expanded'] = data['Review'].apply(contractions.fix)
print(data['Expanded'])

# Remove emojis
data['No_Emojis'] = data['Review'].apply(emoji.demojize)
print(data['No_Emojis'])

# Remove links
data['No_Links'] = data['Review'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
print(data['No_Links'])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Review'])
print(X.toarray())

# Split data into training and testing sets
y = data['Liked']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:')
print(report)

# Function to preprocess a new review
def preprocess_review(review):
    review = review.lower()
    review = BeautifulSoup(review, "html.parser").get_text()
    review = re.sub(f"[{string.punctuation}]", " ", review)
    review = contractions.fix(review)
    review = emoji.demojize(review)
    tokens = word_tokenize(review)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    cleaned_review = ' '.join(lemmatized_tokens)
    return cleaned_review

# Prediction of a new review
new_review = input("Enter a review: ")
cleaned_review = preprocess_review(new_review)
new_review_vectorized = vectorizer.transform([cleaned_review])
prediction = model.predict(new_review_vectorized)
if prediction[0] == 1:
    print("The review is predicted to be positive")
else:
    print("The review is predicted to be negative")