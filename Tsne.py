import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Tsne:
    def __init__(self,ngram_df, w2v_model):
        self.ngram_df = ngram_df
        self.w2v_model = w2v_model


    def visualize_tsne_w2v(self,plot_length,plot_height):
        total_vocab = list(self.w2v_model.wv.vocab)
        self.ngram_df['term'] = self.ngram_df['term'].str.replace(' ', '_')
        ngram_vocab =  self.ngram_df['term'].values.tolist()
        ngram_vocab_keyerror = []
        for x in ngram_vocab:
            if x in total_vocab:
                ngram_vocab_keyerror.append(x)

        ngram_w2v_model = self.w2v_model[ngram_vocab_keyerror]
        tsne = TSNE(n_components=2)
        ngram_tsne = tsne.fit_transform(ngram_w2v_model)

        df = pd.DataFrame(ngram_tsne, index=ngram_vocab_keyerror, columns=['x', 'y'])
        fig = plt.figure(figsize=(plot_length, plot_height))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df['x'], df['y'])
        for word, pos in df.iterrows():
            ax.annotate(word, pos)


