#%% md
# # Sentiment Analysis
#%%
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# !pip install scikit-learn-intelex
# from sklearnex import patch_sklearn 
# patch_sklearn()
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import contractions
from emot.emo_unicode import EMOTICONS_EMO
import emoji
warnings.filterwarnings('ignore')
#%%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#%%
rand_state = 5624 #course code
#%%
data = pd.read_csv('Suicide_Detection.csv')
data.drop(columns=['Unnamed: 0'], inplace=True)
#%% md
# # Initial Cleanup and preprocessing
# - Emojis and emoticons are converted to words
# - URLs and Emails are removed since they dont contribute to semantic relationship.
#%%
data.info()
#%%
def handle_emoticons(txt):
  for emot in EMOTICONS_EMO:
    txt = re.sub(u'('+re.escape(str(emot))+')', "_".join(EMOTICONS_EMO[emot].replace(",","").split()), txt)
  return txt

def cleanup(text):
    # convert to lower case
    text = text.lower()
    #removing fillers and unwanted characters
    text = text.replace('filler', '')
    text = text.replace('\u200b', '')
    text = text.replace('&amp;#x200B;', '')
    # removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # removing email addresses
    text = re.sub(r'\S+@\S+', '', text)
    #handling contractions
    text = contractions.fix(text)
    #handle emojis
    text = emoji.demojize(text)
    # handling emoticons
    text = handle_emoticons(text)
    # removing extra whitespace
    text = ' '.join(text.split())
    
    return text
#%%
data.drop_duplicates(subset='text', keep='first', inplace=True)
data.dropna(subset=['text'], inplace=True)

data['text'] = data['text'].apply(cleanup)
#%%
def process(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    lemma = WordNetLemmatizer()
    tokens = [ lemma.lemmatize(token) for token in tokens if token not in stopwords.words('english') and len(token) > 2 ]
    return ' '.join(tokens)
#%%
data['tokens']  = data['text'].apply(process)
#%%
# saving the preprocessed text, to avoid redundant executions.
data.to_csv('processed_data.csv', index = False)
#%% md
# # Dataset analysis
# 
#%%
def to_string(txt):
    txt = str(txt)
    return txt
#%%
data = pd.read_csv('processed_data.csv')

data['text'] = data['text'].apply(to_string)
data['tokens'] = data['tokens'].apply(to_string)

#%%
data.head()
#%%
data.info()
#%% md
# - No NULL values found in the dataset
#%% md
# ### Checking for class imbalance to see if dataset is skewed
# > There is no imbalance seen in the datapoints for suicide and non-suicide related posts.
#%%
plt.figure(figsize=(10, 6))
class_dist = data['class'].value_counts()
sns.barplot(x=class_dist.index, y=class_dist.values)
plt.title('Class Distribution')
plt.ylabel('Count')
plt.show()
#%%
print(class_dist)
print(f"\nImbalance ratio: {class_dist.max() / class_dist.min():.2f}")
#%% md
# ### Analysing length of the posts
# > Studies also show that people experiencing these thoughts often write more detailed and emotional content.
# 
# > Shorter length texts can also indicate immediate crisis.
#%%
data['text_length'] = data['text'].str.len()
#%%
sns.histplot(data=data, x='text_length', hue='class', bins=30)
plt.title('Text Length by Class')
plt.show()

print("\nText Length Stats:")
print(data.groupby('class')['text_length'].describe())
#%% md
# - Based on the stats, we can observe that suicide related posts are usually longer.
#%% md
# ### Analyzing sentiment of the posts
# > Emotional content maybe more relevant to suicidal posts.
# 
#%%
def sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

data['sentiment'] = data['tokens'].apply(sentiment)
#%%
plt.figure()
sns.boxplot(x='class', y='sentiment', data=data)
plt.title('Sentiment Distribution by Class')
plt.show()
#%%
sns.kdeplot(data=data, x='sentiment', hue='class')
plt.title('Sentiment Density by Class')
plt.tight_layout()
plt.show()
#%% md
# ### Word Clouds
# > To look for patterns and keywords
#%%
plt.figure()

suicide_text = ' '.join(data[data['class'] == 'suicide']['text'])
wordcloud = WordCloud().generate(suicide_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Suicide posts')
plt.show()
#%%
plt.figure()
non_suicide_text = ' '.join(data[data['class'] == 'non-suicide']['text'])
wordcloud = WordCloud().generate(non_suicide_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Non-suicide posts')
plt.show()
#%%
data.to_csv('processed_data.csv')
#%% md
# # Linguistic Inquiry and Word Count
#%%
data = pd.read_csv('processed_data.csv')

data['text'] = data['text'].apply(to_string)
data['tokens'] = data['tokens'].apply(to_string)

#%%
emotional_words = {
    'negative_emotions': {
        'sad', 'depressed', 'lonely', 'hopeless', 'worthless', 'empty', 
        'miserable', 'pain', 'hurt', 'crying', 'tears', 'grief', 'depression',
        'anxiety', 'scared', 'fear', 'worried', 'stress', 'panic', 'anxious', 'hate',
        'numb', 'exhausted', 'tired', 'drained', 'broken', 'defeated', 'darkness',
        'suffering', 'agony', 'despair', 'meaningless', 'rejected', 'abandoned',
        'ashamed', 'guilty', 'failure', 'useless', 'burden', 'trapped', 'alone'
    },
    'positive_emotions': {
        'happy', 'joy', 'excited', 'love', 'wonderful', 'peace', 'peaceful',
        'calm', 'relaxed', 'confident', 'proud', 'strength', 'hope', 'grateful','blessed', 'thankful', 'appreciated', 'supported', 'understood', 'accepted',
        'worthy', 'valuable', 'capable', 'strong', 'motivated', 'inspired',
        'energized', 'optimistic', 'better', 'improving', 'healing', 'recovering'
    },
    'suicidal_thoughts': {
        'suicide', 'die', 'death', 'kill', 'end', 'gone', 'goodbye', 'final',
        'leave', 'sleep', 'forever', 'peace', 'escape', 'free','hopeless', 'helpless', 'pointless', 'meaningless', 'worthless', 'futile',
        'impossible', 'unchangeable', 'unfixable', 'irreparable', 'permanent',
        'endless', 'infinite', 'eternal', 'forever', 'never', 'cannot', 'stuck',
        'trapped', 'imprisoned', 'confined', 'limited', 'restricted'
    },
    'help_seeking': {
        'help', 'support', 'advice', 'please', 'need', 'someone', 'anybody',
        'listen', 'talk', 'therapy', 'counseling', 'professional',
        'hotline', 'crisis', 'emergency', 'hospital', 'doctor', 'psychiatrist',
        'medication', 'treatment', 'appointment', 'diagnosis', 'recovery', 'cope'
    }
}
#%%
def get_liwc_features(text):
    words = text.split()
    total_words = len(words)
    
    category_count = defaultdict(int)
    for word in words:
        for category,word_set in emotional_words.items():
            if word in word_set:
                category_count[category] += 1
                
    features = {}
    for category in emotional_words.keys():
        count = category_count[category]
        percentage = (count / total_words * 100) if total_words > 0 else 0
        features[f'{category}_count'] = count
        features[f'{category}_ratio'] = percentage
    
    features['total_words'] = total_words
    features['unique_words'] = len(set(words))
    
    return features
        
#%%
def combine_features(data,tokens):
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,3), stop_words='english', lowercase=False)
    tfidf_features = tfidf.fit_transform(tokens.astype('U'))
    
    liwc_features = [get_liwc_features(str(text)) for text in tokens]
    liwc_data = pd.DataFrame(liwc_features)
    
    scaler = StandardScaler()
    liwc_scaled = scaler.fit_transform(liwc_data)
    polarity_scaled = scaler.fit_transform(data['sentiment'].values.reshape(-1, 1))
    text_len_scaled = scaler.fit_transform(data['text_length'].values.reshape(-1,1))
    
    combined_features = np.hstack([tfidf_features.toarray(), liwc_scaled, polarity_scaled, text_len_scaled])
    
    return combined_features
#%%
features = combine_features(data,data['tokens'])
#%% md
# #  Running Classifiers
#%%
data['class_code'] = (data['class'] == 'suicide').astype(int)
#%%
x_train, x_test, y_train, y_test = train_test_split(features, data['class_code'], test_size=0.2, random_state=rand_state)
#%%
classifiers = {
    'Random forest' : RandomForestClassifier(n_estimators=100, random_state=rand_state),
    'Decision Tree' : DecisionTreeClassifier(criterion='entropy', random_state=rand_state),
    'SVM' : LinearSVC(random_state=rand_state),
}
#%%
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
                xticklabels =['Non-suicide', 'suicide'],
                yticklabels=['Non-suicide', 'suicide'])
    plt.title(f'Confusion matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#%%
for name, clf in classifiers.items():
    print(f'{name} Classifier')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f'\n\nConfusion matrix for {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print(f'{name} Classification report:\n', classification_report(y_test, y_pred))
    