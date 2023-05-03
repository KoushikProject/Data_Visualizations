#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[126]:


DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('Downloads/archive/twitterdata.csv.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


# In[127]:


df.head(50)


# In[128]:


data=df[['text','target']]
data['target'] = data['target'].replace(4,1)


# In[129]:


data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]


# In[130]:


data_neg.head()


# In[131]:


dataset = pd.concat([data_pos, data_neg])
dataset.head()


# In[132]:


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[133]:


len(stopwordlist)


# In[134]:


from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# In[135]:


en_stopwords = set(stopwords.words("english"))


# In[136]:


len(en_stopwords)


# In[137]:


data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]


# In[138]:


dataset1 = pd.concat([data_pos, data_neg])


# In[139]:


X = dataset1['text']
y = dataset1['target']


# In[140]:


STOPWORDS = set(stopwordlist)
import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import nltk
st = nltk.PorterStemmer()
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
lm = nltk.WordNetLemmatizer()

def getcleanedtext(text):
    text=text.lower()
    def cleaning_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in en_stopwords])
    stop_word_removal = cleaning_stopwords(text)
    
    def cleaning_punctuations(text):
        translator = str.maketrans('', '', punctuations_list)
        return text.translate(translator)
    punctuation_removal = cleaning_punctuations(stop_word_removal)
    
    def cleaning_repeating_char(text):
        return re.sub(r'(.)1+', r'1', text)
    repeating_char_removal = cleaning_repeating_char(punctuation_removal)
    
    def cleaning_URLs(data):
        return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
    clean_url = cleaning_URLs(repeating_char_removal)
    
    def cleaning_numbers(data):
        return re.sub('[0-9]+', '', data)
    clean_numbers = cleaning_numbers(clean_url)
    
    def stemming_on_text(data):
        text = [st.stem(word) for word in data]
        return data
    stemming = stemming_on_text(clean_numbers)
    
    def lemmatizer_on_text(data):
        text = [lm.lemmatize(word) for word in data]
        return data
    lemmatize = lemmatizer_on_text(stemming)
    
    tokens=tokenizer.tokenize(lemmatize)
    clean_text=' '.join(tokens)
    return clean_text


# Function to preprocess Reviews data
#def preprocess_Reviews_data(data,name):
    # Proprocessing the data
#    data[name]=data[name].str.lower()
    # Code to remove the Hashtags from the text
#    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+','',x))
    # Code to remove the links from the text
#    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Code to remove the Special characters from the text 
#    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Code to substitute the multiple spaces with single spaces
#    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
#    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    # Remove the twitter handlers
#    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))


# Function to tokenize and remove the stopwords    
#def rem_stopwords_tokenize(data,name):
      
#    def getting(sen):
#        example_sent = sen

#        filtered_sentence = [] 

#       stop_words = set(stopwords.words('english')) 

#        word_tokens = word_tokenize(example_sent) 
        
#        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
#        return filtered_sentence
    # Using "getting(sen)" function to append edited sentence to data
#    x=[]
#    for i in data[name].values:
#        x.append(getting(i))
#    data[name]=x


# In[141]:


x_clean=[getcleanedtext(i) for i in X]


# In[142]:


x_clean


# In[143]:


dataset1.head()
len(dataset1)


# In[144]:


y


# In[145]:


X_train, X_test, y_train, y_test = train_test_split(x_clean,y,test_size = 0.05, random_state =26105111)


# In[146]:


print(X_train)


# In[147]:


print(X_test)


# In[148]:


print(y_train)


# In[149]:


print(y_test)
outcome = y_test


# # TfidfVectorizer

# In[96]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)


# In[97]:


X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


# In[98]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report


# In[99]:


print(X_test)


# In[100]:


def model_Evaluate(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# In[101]:


BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
y_pred1 = BNBmodel.predict(X_test)


# In[102]:


y_pred1


# In[103]:


outcome_comparison=pd.DataFrame()


# In[104]:


outcome_comparison['actual'] = outcome


# In[105]:


outcome_comparison['predicted']= y_pred1


# In[106]:


outcome_comparison


# In[107]:


model_Evaluate(BNBmodel)


# In[121]:


DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
df1 = pd.read_csv('Desktop/test.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


# In[123]:


df1.head()
xtesting = df1['text']


# In[124]:


xtesting_clean = [getcleanedtext(i) for i in xtesting]


# In[125]:


xtesting_clean


# In[150]:


DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
dfsource = pd.read_csv('Downloads/archive/twitterdata.csv.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


# In[151]:


dfsource.head()


# In[152]:


X = dfsource['text']
y = dfsource['target'].replace(4,1)


# In[153]:


X


# In[154]:


y


# In[155]:


x_clean = [getcleanedtext(i) for i in X]


# In[156]:


x_clean


# In[158]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(x_clean)


# In[160]:


x_vec = vectoriser.transform(x_clean)


# In[161]:


x_vec


# In[162]:


xt_vec = vectoriser.transform(xtesting_clean)


# In[163]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report


# In[166]:


BNBmodel = BernoulliNB()
BNBmodel.fit(x_vec, y)
y_pred1 = BNBmodel.predict(xt_vec)


# In[167]:


y_pred1


# In[175]:


outcome_comparison1=pd.DataFrame()
outcome_comparison1['actual'] = df1['target'].replace(4,1)
outcome_comparison1['predicted']= y_pred1


# In[179]:


outcome_comparison1

