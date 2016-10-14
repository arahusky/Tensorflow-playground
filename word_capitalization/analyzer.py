# -*- coding: utf-8 -*-
import numpy as np
import os
import nltk
import matplotlib.pyplot as plt

text = ''
topic_folders = os.listdir('bbc')
for topic_folder in topic_folders:
    if os.path.isdir(os.path.join('bbc', topic_folder)):  # consider only folders
        # print('Loading topic:' + topic_folder)
        topic_texts = os.listdir(os.path.join('bbc', topic_folder))

        for topic_text in topic_texts:
            # print(topic_text)
            with open(os.path.join('bbc', topic_folder, topic_text), 'r') as reader:
                # skip first line (heading)
                reader.readline()
                line = reader.readline().decode("ascii", "ignore").encode("ascii")
                while line:
                    if line.strip() is not '':
                        line = line.strip()
                        # line = re.sub(r'\.([a-z]+)', r'. \1', line)
                        text += " " + line
                    line = reader.readline().decode("ascii", "ignore").encode("ascii")

print('Text loaded')
sentences = nltk.sent_tokenize(text)
print('Sentences parsed')
lengths = [len(nltk.word_tokenize(sentence)) for i,sentence in enumerate(sentences)]
print('Showin hist')
a, b, c = plt.hist(lengths, bins=30)
plt.show()

print(type(a))
print(type(b))
print(a)
print(b)
print(c)