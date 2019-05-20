from DifferentiatorySequenceMining import TFISFBasedMining
import pandas as pd
import collections

exp_dir = "../experiments"
data_dir = "../data"

res_dict = collections.defaultdict(list)
for k in range(2, 6):
    print(k)
    seqs_file = "%s/sequences.csv" % data_dir
    name_column = 'name_h'
    sequences = pd.read_csv(seqs_file)
    sequences.set_index(name_column, drop=True, inplace=True)
    tfisf = TFISFBasedMining.TFISF()
    tfisf.extract_kgrams(sequences, k)
    vectors = tfisf.vectorization(sequences, freq_damping=False, use_idf=True, vector_len=None)
    silhouette_score = tfisf.silhouette_score("%s/group1.csv" % data_dir, "%s/group2.csv" % data_dir)
    print("Silhouette score is", silhouette_score)
