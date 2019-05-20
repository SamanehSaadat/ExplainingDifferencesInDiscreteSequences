import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from DifferentiatorySequenceMining import TFISFBasedMining
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *

# fix random seed for reproducibility
np.random.seed(7)

exp_dir = "../experiments"
data_dir = "../data"
words_count = 14

t = pd.read_csv("%s/group1.csv" % data_dir)
hb_teams = t[t.columns[0]].tolist()
t = pd.read_csv("%s/group2.csv" % data_dir)
h_teams = t[t.columns[0]].tolist()
seqs = pd.read_csv("%s/sequences.csv" % exp_dir).set_index('name_h')
seqs['label'] = np.where(seqs.index.isin(hb_teams), 1, -1)
seqs['label'] = np.where(seqs.index.isin(h_teams), 0, seqs['label'])
seqs = seqs[seqs['label'] > -1]
print(seqs.shape)

########### LSTM ########################
def to_vec(x):
    return np.array([e for e in list(x)])

seqs['vector'] = seqs['sequence'].apply(to_vec)
X_train, X_test, y_train, y_test = train_test_split(seqs['vector'], seqs['label'], test_size=0.2)

# truncate and pad input sequences
max_seq_length = 1000
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(words_count, embedding_vecor_length, input_length=max_seq_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


###### SUPPORT VECTOR MACHINE ###########
sl = TFISFBasedMining.TFISF()
k = 2
sl.extract_kgrams(seqs, k)
sl.vectorization(seqs, freq_damping=False, use_idf=True, vector_len=None)
s, X, y = sl.silhouette_score(seqs, "%s/group1.csv" % data_dir, "%s/group2.csv" % data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
