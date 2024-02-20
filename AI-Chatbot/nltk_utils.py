import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

'''
This file is used as a sort of helper file to help tokenize the sentences and create stems of the words. It also will create the 'bag of words'
which essentially is what will later on be used to train the neural net model
'''

#tokenize the sentence and stem the words 
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

# essentially creates connections betweeen a tokenized sentence and the all the words by marking them in arr as 1
# eg. sentence = ['hello', 'how', 'are', 'you']
# words = ['howdy', 'lets', 'how', 'you']
# bag = [0, 0, 1, 1] because how and you are present in sentence
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag
