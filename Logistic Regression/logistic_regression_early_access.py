#Logistic Regression for Early Access (English, German, Romanian)

import json_lines
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from nltk.stem import PorterStemmer
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer

from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0

def plot_roc_curve(fpr, tpr, title, num, curve):
    plt.figure(num)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + str(title))
    plt.legend([curve])
    plt.show()
    
corpus=[[],[]];y=[];
with open('file_53.jl','rb') as f:
    for item in json_lines.reader(f):
        try:
            if(detect(item['text']) == "en" or detect(item['text']) == "ro"
               or detect(item['text']) == "de"
               ):
                corpus[0].append(item['text'])
                corpus[1].append(detect(item['text']))
                y.append(item['early_access'])
        except:
            print("Cannot recognize language, or text has no features") 
corpus = np.array(corpus)
corpus = np.transpose(corpus)
y = np.array(y)

early_access_false = 0
early_access_true = 0
        
for i in y:
    if i == True:
        early_access_true += 1
    else:
        early_access_false += 1


plt.figure(1)
plt.bar(['True', 'False'],[early_access_true, early_access_false], color=['green','red'])
plt.xlabel("True/False")
plt.ylabel("Number")
plt.title("Early Access Distribution")
plt.show()

tokenizer = TfidfVectorizer().build_tokenizer()
xtrain_1=[]
stemmer1 = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer() 
for i in range(len(corpus)):
    if(corpus[i][1] == "en"):
        stemmer2 = SnowballStemmer("english")
    if(corpus[i][1] == "de"):
        stemmer2 = SnowballStemmer("german")
    if(corpus[i][1] == "ro"):
        stemmer2 = SnowballStemmer("romanian")     
    
    X = str(corpus[i]).lower()
    Y=str(X).replace('\n','')
    X=tokenizer(X)
    X=WhitespaceTokenizer().tokenize(str(X))
    X=word_tokenize(str(X))    
    tokens = word_tokenize(str(X))
    stems = [stemmer2.stem(token) for token in tokens]
    xtrain_1.append(str(stems))


stopwords_english = set(stopwords.words('english'))
stopwords_german = set(stopwords.words('german'))
stopwords_romanian = set(stopwords.words('romanian'))


final_stopwords = set()
final_stopwords = final_stopwords.union(stopwords_english, stopwords_romanian, 
                                        stopwords_german)

tfid = TfidfVectorizer(stop_words=final_stopwords, max_df=0.2)
X = tfid.fit_transform(xtrain_1)    

kf = KFold(n_splits = 5)
accuracy = []
std = []
C_range = [0.001, 0.01, 0.1, 1, 10, 100]
for Ci in C_range:
    temp = []   
    temp_pre = []
    model = LogisticRegression(penalty = 'l2', C = Ci, max_iter = 10000000)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        temp.append(metrics.accuracy_score(y[test], y_pred))
    accuracy.append(np.array(temp).mean())
    std.append(np.array(temp).std())
   
plt.figure(2)
plt.errorbar(C_range,accuracy,std)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('C vs Accuracy')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = LogisticRegression(penalty = 'l2', C = 10, max_iter = 10000000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Logistic Regression)', 3, 'Logistic Regression')

cm = confusion_matrix(y_test, preds) 
print(classification_report(y_test, preds))
print("Accuracy:",metrics.accuracy_score(y_test, preds))
print("Precision:",metrics.precision_score(y_test, preds))
print("Recall:",metrics.recall_score(y_test, preds))

from sklearn.dummy import DummyClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)

probs = dummy_clf.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Baseline)', 4, 'Baseline')
cm_baseline = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred))
print("Accuracy Baseline:",metrics.accuracy_score(y_test, y_pred))
print("Precision Baseline:",metrics.precision_score(y_test, y_pred))
print("Recall Baseline:",metrics.recall_score(y_test, y_pred))