import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
class Preprocessor:
    def __init__(self, language):
        self.language = language

    def read_text_files(self,folder_path):
        new_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        text = f.read()
                        new_list.append(text)
        return new_list

    def most_frequent_ngrams(self,corpus,ngram_start,ngram_end, top_n):
        list_ngrams =[]
        c_vec = CountVectorizer(ngram_range=(ngram_start, ngram_end), stop_words=self.language)
        ngrams = c_vec.fit_transform(corpus)
        vocab = c_vec.vocabulary_
        count_values = ngrams.toarray().sum(axis=0)
        for ng_count, ng_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
            list_ngrams.append([ng_text,ng_count])
        df_ngram = pd.DataFrame(list_ngrams,columns=['term','frequency'])
        df_top_n =  df_ngram.head(top_n)
        return df_top_n