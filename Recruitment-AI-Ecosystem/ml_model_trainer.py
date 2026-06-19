import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


print("📖 Loading resume dataset...")
df = pd.read_csv("UpdatedResumeDataSet.csv")

print(f"📊 Dataset loaded successfully! Total records found: {len(df)}")
print(f"🗂️ Unique job categories available: {df['Category'].nunique()}")

def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)  
    text = re.sub(r'RT|cc', ' ', text)      
    text = re.sub(r'#\S+', ' ', text)       
    text = re.sub(r'@\S+', ' ', text)     
    text = re.sub(r'[^\w\s]', ' ', text)    
    text = text.lower()                     
    return ' '.join(text.split())           

print("🧼 Cleaning raw text data (this takes a few seconds)...")
df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

X = df["Cleaned_Resume"]
Y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("🤝 Vectorizing text strings using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("🧠 Training the Machine Learning Classifier...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\n🚀 --- TRAINING COMPLETE --- 🚀")
print(f"🎯 Model Accuracy on Unseen Test Data: {accuracy * 100:.2f}%")


import pickle


print("💾 Saving ML artifacts for production app utilization...")


with open("nb_resume_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)


with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✨ Successfully generated 'nb_resume_model.pkl' and 'tfidf_vectorizer.pkl'!")