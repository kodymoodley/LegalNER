import gensim
import nltk
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases
import re

class Word_embeddings:
    def __init__(self,corpus,language): #corpus on which word2vec has to be trained
        self.language = language
        self.corpus =corpus


    def pre_processing(self, corpus):  # corpus is a list of reviews
        text_corpus = []
        for i in range(len(corpus)):

            text_corpus.append(corpus[i])
            text_corpus[i] = [sentence for sentence in self.sentence_tokenizer(text_corpus[i])]  # sentence tokenizer
            text_corpus[i] = [self.remove_words_with_specialcharacs(item) for item in
                              text_corpus[i]]  # remove hastags,mentions@
            text_corpus[i] = [self.word_tokenizer(sentence) for sentence in text_corpus[i]]  # word tokenizer
            text_corpus[i] = [item for sublist in text_corpus[i] for item in sublist]  # flatten the list text_corpus
            text_corpus[i] = [item for item in text_corpus[i] if
                              item.isalpha()]  # taking strings with only alphabetic characters
        # text_corpus[i] = [item for item in text_corpus[i] if len(item) > 1] #remove single character words like 'a'
        # print(text_corpus)
        return text_corpus

    def remove_words_with_specialcharacs(self, text):  # text is a string of words, like text = "Today I am happy."
        text = re.sub('http\S+\s*', '', text)  # remove URLs
        # text = re.sub('RT|cc', '', text)  # remove RT and cc
        text = re.sub('#\S+', '', text)  # remove hashtags
        text = re.sub('@\S+', '', text)  # remove mentions
        text = re.sub('`', "'", text)  # substitute similar apostorphe
        text = re.sub('[%s]' % re.escape("""!"#$%&()*+,.:;<=>?@[]^_{|}~"""), '', text)
        text = re.sub('\s+', ' ', text)  # remove extra whitespace
        return text

    def sentence_tokenizer(self, text):
        return nltk.sent_tokenize(text)

    def word_tokenizer(self, text):
        return nltk.word_tokenize(
            text.lower())  # all outputs from word2vec are lowercase, so entities in the table will also be in lowecase

    def stopwords(self):
        stopwds = stopwords.words(self.language)  # NLTK's default stopword list
        return stopwds

    def ngram_generation(self, min_count,threshold):  # using gensim inbuilt function to generate n-grams automatically based on co-occurence
        sent_stream = self.pre_processing(self.corpus)
        bigram = Phrases(sent_stream, min_count, threshold=threshold)
        trigram = Phrases(bigram[sent_stream], min_count, threshold=threshold)
        return [trigram[bigram[doc]] for doc in sent_stream]

    def train_save_model(self,  path, model_name,no_epochs, min_count_ngram=5, threshold_ngram=20, net_size=80):
        """
        :param min_count_ngram: Ignore all words and bigrams with total collected count lower than this value
        :param threshold_ngram: Represent a score threshold for forming the phrases (higher means fewer phrases)
        """
        l = self.ngram_generation(min_count_ngram, threshold_ngram)
        model = gensim.models.Word2Vec(l, size=net_size, window=7, min_count=5, workers=10)
        model.train(l, total_examples=len(l), epochs=no_epochs)  # better set epochs to 20, 20 used in saved model
        model.save(path + model_name)  # original model saved as model.save("word2vec.model"),so use different name

    def load_model(self, model_name):
        model = gensim.models.Word2Vec.load(model_name)
        return model

    def most_similar_top_k_terms(self, term, k,loaded_model):  # term can be a word or phrase (phrase must have underscore between words as it is gensim format)
          # only terms present in the dataset can be searched. Other terms will give KeyError
        return loaded_model.wv.most_similar(positive=term, topn=k)

    def most_similar_top_k_with_frequency(self, term, k, model_name,min_frequency):  # print most_similar terms for a given word or n-gram with more frequency than input
        model = self.load_model(model_name)
        list_words_similarity = model.wv.most_similar(positive=term, topn=k)
        l = []
        for word, sim in list_words_similarity:
            if model.wv.vocab[word].count >= min_frequency:
                l.append((word, model.wv.vocab[word].count))
        return l






