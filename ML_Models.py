#%%
import pandas as pd
import contractions
import re
import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
#%%
df = pd.read_csv('Suicide_Detection.csv')
#%%
df.head()
#%%
df = df.drop(columns=["Unnamed: 0"])
#%%
df
#%%
df.info()
#%%
df = df.dropna(subset=['text'])
df = df.reset_index(drop=True)
#%%
df['class'].nunique()
#%%
df['class'].value_counts()
#%%
df['text'] = df['text'].str.lower()
#%%
url_pattern = r'http[s]?://[^\s]+'
urls_found = df[df['text'].str.contains(url_pattern, regex=True)]

print("Sentences with URLs:")
print(urls_found[['text']])
#%%
df['text'] = df['text'].str.replace(url_pattern, '', regex=True)
#%%
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
emails_found = df[df['text'].str.contains(email_pattern, regex=True)]

print("\nSentences with email addresses:")
print(emails_found[['text']])
#%%
df['text'] = df['text'].str.replace(email_pattern, '', regex=True)
#%%
hashtag_pattern = r'#[\w]+'
hashtags_found = df[df['text'].str.contains(hashtag_pattern, regex=True)]

print("\nSentences with hashtags:")
print(hashtags_found[['text']])
#%%
df['text'] = df['text'].str.replace(hashtag_pattern, '', regex=True)
#%%
df['text'] = df['text'].str.strip().str.replace('\s+', ' ', regex=True)
#%%
df['text'] = df['text'].apply(contractions.fix)
#%%
def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

df['text'] = df['text'].apply(remove_special_characters)
#%%
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

df['text'] = df['text'].apply(remove_numbers)
#%%
df['text'] = df['text'].str.strip().str.replace('\s+', ' ', regex=True)
#%%
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
#%%
def preprocess(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    pos_tags = pos_tag(filtered_tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_tokens)
#%%
df['text'] = df['text'].apply(preprocess)
#%%
model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
#%%
df['SentenceEmbedding'] = df['text'].apply(lambda x: model.encode(x))
#%%
df['SentenceEmbedding']
#%%
X_stacked = np.vstack(df['SentenceEmbedding'])
#%%
X_stacked
#%%
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['class'])
#%%
X_train, X_test, y_train, y_test = train_test_split(X_stacked, encoded_labels, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
#%%
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=5805)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
svc_classifier = SVC(kernel='rbf',random_state=5805)
svc_classifier.fit(X_train, y_train)

y_pred = svc_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
decison_tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=5805)
decison_tree_classifier.fit(X_train, y_train)

y_pred = decison_tree_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))