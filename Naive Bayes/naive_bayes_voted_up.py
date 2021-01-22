from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn import metrics
import json_lines
from nltk.corpus import stopwords
import nltk
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
                y.append(item['voted_up'])
        except:
            continue 
corpus = np.array(corpus)
corpus = np.transpose(corpus)
y = np.array(y)

voted_up_false = 0
voted_up_true = 0

for i in y:
    if i == True:
        voted_up_true += 1
    else:
        voted_up_false += 1

plt.figure(1)
plt.bar(['True', 'False'],[voted_up_true, voted_up_false], color=['green','red'])
plt.xlabel("True/False")
plt.ylabel("Number")
plt.title("Voted Up Distribution")
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

kf = KFold(n_splits = 10)
accuracy = []
temp = []   
std = []
model = BernoulliNB()
for train, test in kf.split(X):
    model.fit(X[train], y[train])
    preds = model.predict(X[test])
    temp.append(metrics.accuracy_score(y[test], preds))
    accuracy.append(np.array(temp).mean()) 
    std.append(np.array(temp).std())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = BernoulliNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)

plt.figure(2)
plt.errorbar(range(0,10),accuracy,std)
plt.xlabel('Folds')
plt.ylabel('Accuracy')
plt.title('Folds vs Accuracy')
plt.show()

probs = model.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr, '(Naive Bayes)', 1, 'Naive Bayes')

cm = confusion_matrix(y_test, preds) 
print(classification_report(y_test, preds))
print("Accuracy Naive Bayes:",metrics.accuracy_score(y_test, preds))
print("Precision Naive Bayes:",metrics.precision_score(y_test, preds, zero_division = 0))
print("Recall Naive Bayes:",metrics.recall_score(y_test, preds, zero_division = 0))

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
plot_roc_curve(fpr, tpr, '(Baseline)', 2, 'Baseline')
cm_baseline = confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred))
print("Accuracy Baseline:",metrics.accuracy_score(y_test, y_pred))
print("Precision Baseline:",metrics.precision_score(y_test, y_pred))
print("Recall Baseline:",metrics.recall_score(y_test, y_pred))

