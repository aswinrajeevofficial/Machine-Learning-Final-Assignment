import json_lines
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from nltk.stem.snowball import SnowballStemmer
from langdetect import DetectorFactory, detect
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
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
C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
for Ci in C_range:
    model = LinearSVC(C = Ci, max_iter = 1000000)
    temp = []
    temp_train = []
    plotted = False
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        temp.append(metrics.accuracy_score(y[test], y_pred))
    accuracy.append(np.array(temp).mean()) 
    std.append(np.array(temp).std())
plot5 = plt.figure(2)
plt.title('Accuracy vs C')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.errorbar(C_range,accuracy,std)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
model = LinearSVC(C=0.1, max_iter = 1000000)
model.fit(X_train, y_train)
preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds) 
print(classification_report(y_test, preds))
print("Accuracy:",metrics.accuracy_score(y_test, preds))
print("Precision:",metrics.precision_score(y_test, preds))
print("Recall:",metrics.recall_score(y_test, preds))

auc = roc_auc_score(y_test, model.decision_function(X_test))
print('AUC: %.2f' % auc)
fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
plot_roc_curve(fpr, tpr, '(SVM)', 3, 'SVM')