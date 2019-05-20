import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import silhouette_score


class TFISF:
    def extract_kgrams(self, sequences, k):
        sequences['k_gram'] = sequences['sequence'].apply(lambda x: self.kgrams(x, k))

    def kgrams(self, s, k):
        k_grams = []
        for i in range(len(s) - k):
            k_grams.append(s[i: i + k])
        return k_grams

    def vectorization(self, sequences, freq_damping, use_idf, vector_len):
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=freq_damping, use_idf=use_idf, max_features=vector_len)
        X = sequences['k_gram'].tolist()
        X = [" ".join(s) for s in X]
        v = tfidf_vectorizer.fit_transform(X)
        vectors = pd.DataFrame(v.toarray(),
                                    columns=tfidf_vectorizer.get_feature_names(),
                                    index=sequences.index)
        return vectors

    def count_vectorization(self, sequences, vector_len, is_binary):
        vectorizer = CountVectorizer(max_features=vector_len, binary=is_binary)
        X = sequences['k_gram'].tolist()
        X = [" ".join(s) for s in X]
        v = vectorizer.fit_transform(X)
        vectors = pd.DataFrame(v.toarray(),
                                    columns=vectorizer.get_feature_names(),
                                    index=sequences.index)
        return vectors

    def get_top_n_words(self, sequences, group_file, n=None):
        """
        List the top n words in a vocabulary according to occurrence in a text corpus.
        """
        group = pd.read_csv(group_file)
        group = group[group.columns[0]].tolist()
        X = sequences[sequences.index.isin(group)]['k_gram'].tolist()
        X = [" ".join(s) for s in X]
        vec = CountVectorizer().fit(X)
        bag_of_words = vec.transform(X)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def silhouette_score(self, vectors, group1_file, group2_file):
        group1 = pd.read_csv(group1_file)
        group1 = group1[group1.columns[0]].tolist()
        group2 = pd.read_csv(group2_file)
        group2 = group2[group2.columns[0]].tolist()
        vectors['label'] = np.where(vectors.index.isin(group1), 1, 0)
        vectors['label'] = np.where(vectors.index.isin(group2), 2, vectors['label'])
        X = vectors[vectors['label'] > 0]
        y = X['label']
        X.drop('label', axis=1, inplace=True)
        return silhouette_score(X, y, metric='cosine')

