#%%
import string
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pprint import pprint
from bertopic import BERTopic
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#%%
pip install xgboost
#%%
file = "Suicide_Detection.csv"
df = pd.read_csv(file)
df.head(5)
#%%
df.drop("Unnamed: 0", axis=1, inplace=True)
#%%
df.head(5)
#%%
df["class"].value_counts()
#%%
def na_perc(df):
    return (df.isnull().sum() / len(df)) * 100
#%%
na_perc(df)
#%%
df.info()
#%%
df.shape
#%%
def preprocess(s):
    s = str(s)
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s
#%%
df = df.map(preprocess)
df.head(5)
#%%
def load(file):
    mydic = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.split()
            mydic[line[0]] = np.array(line[1:], dtype="float32")
    return mydic
#%%
glove = load("glove.6B.300d.txt")
#%%
def avg_glove(glove, document):
    res = np.zeros(300)
    words = 0
    document = document.split()
    for word in document:
        if(word in glove):
            res += glove[word]
            words += 1
    if(words == 0):
        return res
    else:
        return res/words
#%%
data = df["text"]
#%%
x_glove = [avg_glove(glove, d) for d in data]
#%%
long_string = ','.join(list(df['text'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()
#%%
bert_data = df["text"].tolist()
bert_data[:5]
#%%
bert_score = []
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(bert_data)
#%%
df["topic"] = topics
df.head(5)
#%%
df["glove_vect"] = x_glove
df.head(5)
#%%
x_glove
#%%
df["class"].unique()
#%%
df["class"] = df["class"].map({'suicide': 0, 'nonsuicide': 1})
df.head(5)
#%%
df.info()
#%%
type(x_glove)
#%%
topics = np.array(topics)
topics = topics.reshape(-1, 1)
x_with_topic = np.hstack((x_glove, topics))
#%%
#x_with_topic = [x_glove, topics]
x_no_topic = x_glove
y = df["class"]
#%%
df.isnull().sum()
#%%
y
#%% md
# Decision Tree
#%%
kf = KFold(n_splits=10, shuffle=True, random_state=1)
#%%
type(y)
#%%
x_no_topic = np.array(x_no_topic)
#%%
precisions, recalls, accuracies, f1_scores = [], [], [], []
clf = DecisionTreeClassifier(random_state=1)

for train_index, test_index in kf.split(x_no_topic):
    x_train, x_test = x_no_topic[train_index], x_no_topic[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
#%%
precisions, recalls, accuracies, f1_scores = [], [], [], []
clf = DecisionTreeClassifier(random_state=1)

for train_index, test_index in kf.split(x_with_topic):
    x_train, x_test = x_with_topic[train_index], x_with_topic[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
#%% md
# SVM
#%%
precisions, recalls, accuracies, f1_scores = [], [], [], []
svm = SVC(random_state=1)

for train_index, test_index in kf.split(x_no_topic):
    x_train, x_test = x_no_topic[train_index], x_no_topic[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svm.fit(x_train, y_train)
    
    y_pred = svm.predict(x_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
#%%
precisions, recalls, accuracies, f1_scores = [], [], [], []
svm = SVC(random_state=1)

for train_index, test_index in kf.split(x_with_topic):
    x_train, x_test = x_with_topic[train_index], x_with_topic[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svm.fit(x_train, y_train)
    
    y_pred = svm.predict(x_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    accuracies.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
#%%
# Random Forest Classifier - No Topic
#%%
X_train, X_test, y_train, y_test = train_test_split(x_no_topic, y, test_size=0.2, random_state=42)
#%%
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=5805)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
## Support Vector Machine - No Topic
#%%
svc_classifier = SVC(random_state=5805)
svc_classifier.fit(X_train, y_train)

y_pred = svc_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
## Decision Tree Classifier - No Topic
#%%
decison_tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=5805)
decison_tree_classifier.fit(X_train, y_train)

y_pred = decison_tree_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
# Random Forest Classifier - With Topic
#%%
X_train, X_test, y_train, y_test = train_test_split(x_with_topic, y, test_size=0.2, random_state=42)
#%%
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=5805)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
## Support Vector Machine - With Topic
#%%
svc_classifier = SVC(random_state=5805)
svc_classifier.fit(X_train, y_train)

y_pred = svc_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))
#%%
## Decision Tree Classifier - With Topic
#%%
decison_tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=5805)
decison_tree_classifier.fit(X_train, y_train)

y_pred = decison_tree_classifier.predict(X_test)
#%%
print("Classification Report:")
print(classification_report(y_test, y_pred))