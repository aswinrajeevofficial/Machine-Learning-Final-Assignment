import json_lines
import numpy as np
from langdetect import DetectorFactory, detect
DetectorFactory.seed = 0

corpus=[];y=[];z=[]
with open('file_53.jl','rb') as f:
    for item in json_lines.reader(f):
        corpus.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

corpus = np.array(corpus)
y = np.array(y)
z = np.array(z)

f = open("languages.txt", "w")

for i in corpus:
    try:
        f.write(detect(i) + '\n')
    except:
        f.write('Error\n')
f.close()

