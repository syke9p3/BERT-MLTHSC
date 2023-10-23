import io
import re
from collections import Counter
import pickle 

def read_file(path):
    text = ""
    with io.open(path, 'r', encoding='ISO-8859-1') as f:
        text = f.read()
    
    return text

def get_words(plain):
    doc_words = []
    clean = re.sub('[^a-zA-Z]', ' ', plain.lower())
    doc_words = clean.split()
    
    return doc_words

path = "../dataset/bigdata.txt"
plain = read_file(path)
doc_words = get_words(plain)
word_freq = Counter(doc_words)

class SpellChecker:
    
    def __init__(self, word_freq):
        self.w_rank = {}
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        
        N = sum(word_freq.values())
        for term in word_freq:
            self.w_rank[term] = word_freq[term] / N
    
    def P(self, word): 
        return self.w_rank.get(word, 0)

    def known(self, words): 
        return set(w for w in words if w in self.w_rank)

    def edits1(self, word):
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in self.letters]
        inserts    = [L + c + R               for L, R in splits for c in self.letters]
        
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def correction(self, word):
        return max(self.candidates(word), key = self.P)
    
    def candidates(self, word): 
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

spellchecker_model = SpellChecker(word_freq)
# print(spellchecker_model.correction('Eglisia'))

with open('../model/spellchecker_model.pkl', 'wb') as model_file:
    pickle.dump(spellchecker_model, model_file)