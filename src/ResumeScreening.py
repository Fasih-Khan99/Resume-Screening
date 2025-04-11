import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("Updated_Computer_Science_Dataset_Realistic.csv")

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Combine relevant columns into a single text column
df['combined_text'] = df['Experience'].astype(str) + ' ' + df['Skill'] + ' ' + df['Qualification']
df['combined_text'] = df['combined_text'].apply(preprocess_text)

# Define and preprocess job requirements
job_requirements = "10+ years, Master's degree in Computer Science, strong skills in React"
job_requirements = preprocess_text(job_requirements)

# Vectorize text data
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
job_vector = vectorizer.transform([job_requirements])

# Compute cosine similarity
cosine_similarities = cosine_similarity(job_vector, tfidf_matrix).flatten()
df['similarity_score'] = cosine_similarities
df = df.sort_values(by='similarity_score', ascending=False)

# Show top candidates
top_candidates = df.head(10)
print(top_candidates[['Candidate', 'Email', 'similarity_score']])

# ------------------------------
# Regression & Classification
# ------------------------------

# Load the dataset again for modeling
data = pd.read_csv('Updated_Computer_Science_Dataset_Realistic.csv')

# 🔧 Fix: Clean the 'Experience' column
def clean_experience(exp):
    if pd.isnull(exp):
        return 0
    match = re.search(r'\d+', str(exp))
    return int(match.group()) if match else 0

data['Experience'] = data['Experience'].apply(clean_experience)

# Map categorical values
data['Qualification'] = data['Qualification'].map({"Bachelor's": 0, "Master's": 1, "PhD": 2})
data['Gender'] = data['Gender'].map({"Male": 0, "Female": 1})

# Regression: Predicting experience
X_reg = data[['Skill', 'Qualification', 'Gender']]
X_reg = pd.get_dummies(X_reg, columns=['Skill'], drop_first=True)
y_reg = data['Experience']

# Classification: Predicting qualification
X_clf = data[['Experience', 'Gender']]
y_clf = data['Qualification']

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.7, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.7, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Linear Regression Mean Squared Error: {mse}")

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
y_pred_clf = log_reg.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Logistic Regression Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
print("Confusion Matrix:")
print(conf_matrix)