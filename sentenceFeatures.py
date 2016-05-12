from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import skipgrams
import math

def combinations(n, k):
    f = math.factorial
    return f(n) / (f(k) * f(n - k))

# Comprising sage advice from:
# http://www.kozareva.com/papers/fintalKozareva.pdf
# http://web.science.mq.edu.au/~rdale/publications/papers/2006/swan-final.pdf
def compute(sentence1, sentence2):
    features = [0] * 7
    tokenizer = RegexpTokenizer(r'\w+')
    words1 = tokenizer.tokenize(sentence1)
    words2 = tokenizer.tokenize(sentence2)
    n = len(words1)
    m = len(words2)

    # word overlap features
    count = 0 # num of same words in sentence
    for word1 in words1:
        for word2 in words2:
            if word1 == word2:
                count += 1

    # TODO: Make it symmetric (improvement?)
    features[0] = count / n # "precision"
    features[1] = count / m # "recall"

    features[2] = sentence_bleu([sentence1], sentence2)
    features[3] = sentence_bleu([sentence2], sentence1)

    # Obtain pairs of adjacent words
    skipgrams1 = skipgrams(words1, 2, 0)
    skipgrams2 = skipgrams(words2, 2, 0)

    count = 0
    for gram1 in skipgrams1:
        for gram2 in skipgrams2:
            if gram1 == gram2:
                count += 1

    features[4] = count / combinations(n, count)
    features[5] = count / combinations(m, count)


    if (n > m):
        features[6] = m / n
    else:
        features[6] = n / m

    return features

