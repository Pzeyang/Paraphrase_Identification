from __future__ import division
from nltk.tokenize import RegexpTokenizer
import pickle

tokenizer = RegexpTokenizer(r'\w+')
paraphraseMap = {}
notParaphrMap = {}
allOccurrenceMap = {}

def fileParse (fileName, rows):
    f = open(fileName, "r")
    f.readline()
    for i in range(0,rows):
        tokens = f.readline().strip().split('\t')
        sentence1 = tokenizer.tokenize(tokens[3].lower())
        sentence2 = tokenizer.tokenize(tokens[4].lower())
        paraphrase = int(tokens[0])

        for word2 in sentence2:
            allOccurrenceMap[word2] = 1.0 + (allOccurrenceMap[word2] if word2 in allOccurrenceMap else 0)
            for word1 in sentence1:
                if (word2 == word1):
                    if (paraphrase == 0):
                        notParaphrMap[word2] = 1.0 + (notParaphrMap[word2] if word2 in notParaphrMap else 0)
                    else:
                        paraphraseMap[word2] = 1.0 + (paraphraseMap[word2] if word2 in paraphraseMap else 0)
                    break
    f.close()

fileParse("msr_paraphrase_train.txt",4076)
fileParse("msr_paraphrase_test.txt",1725)
for word in paraphraseMap:
    paraphraseMap[word] /= allOccurrenceMap[word]
for word in notParaphrMap:
    notParaphrMap[word] /= allOccurrenceMap[word]

pickle.dump(paraphraseMap, open("paraphraseMap", "wb"))
pickle.dump(notParaphrMap, open("notParaphrMap", "wb"))
