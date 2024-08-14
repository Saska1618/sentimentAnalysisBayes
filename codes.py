from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re

class sentimentAnalyzer:
    def __init__(self):
        print("sentimentAnalyzer created")

        self.pos = {} # positive scores for each token
        self.neg = {} # negative scores for each token

        self.all_pos_count = 0
        self.all_neg_count = 0

    def clean_data_part(self, data):
        if data is None:
            raise ValueError
        
        part = data.lower()

        # further cleaning data, so i add the following:

        # remove hyperlinks    
        part = re.sub(r'https?://[^\s\n\r]+', '', part)
            
        # remove HTML markup    
        part = re.sub(r'<\w*/?>', '', part)
        
        # remove hashtags sign
        part = re.sub(r'#', '', part)

        # remove multiple dots
        part = re.sub(r'[.,]+', '.', part)

        toks = word_tokenize(part)
        stop_words = set(stopwords.words('english'))

        filtered = [t for t in toks if t not in stop_words]
        
        return filtered
    


    def train(self, dataset):
        if dataset is None:
            raise ValueError
        
        for i in tqdm(range(len(dataset)), desc="Training"):

            part = dataset[i]

            label = part['label']
            text = part['text']
            clean_text = self.clean_data_part(text)

            for token in clean_text:
                if label == 1:
                    self.all_pos_count += 1
                    if token in self.pos.keys():
                        self.pos[token] += 1
                    else:
                        self.pos[token] = 1

                else:
                    self.all_neg_count += 1
                    if token in self.neg.keys():
                        self.neg[token] += 1
                    else:
                        self.neg[token] = 1

    def test(self, dataset):
        if dataset is None:
            raise ValueError
        
        corrects = 0
        all_data = len(dataset)
        
        for i in tqdm(range(len(dataset)), desc="Testing:"):

            part = dataset[i]
            
            label = part['label']
            text = part['text']

            p, n = self.analyze(text)

            pred = 0
            if p > n:
                pred = 1

            if label == pred:
                corrects += 1

        print(f"Accuracy: {corrects / all_data}\n{corrects} out of {all_data}")


    def analyze(self, text):
        text = self.clean_data_part(text)

        pos_score = self.all_pos_count / (self.all_pos_count + self.all_neg_count)
        neg_score = self.all_neg_count / (self.all_pos_count + self.all_neg_count)

        for token in text:
            if token in self.pos.keys():
                pos_score *= (self.pos[token] / len(self.pos))
            
            if token in self.neg.keys():
                neg_score *= (self.neg[token] / len(self.neg))

        pos_score *= (1 / (self.all_pos_count + self.all_neg_count))
        neg_score *= (1 / (self.all_pos_count + self.all_neg_count))

        return pos_score, neg_score


        
        

