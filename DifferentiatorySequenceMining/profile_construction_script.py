from DifferentiatorySequenceMining import MatrixProfileBasedMining
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

data_dir = "../sample_data"

seqs_file = "%s/sequences.csv" % data_dir
seqs = pd.read_csv(seqs_file)['sequence'].tolist()

m = MatrixProfileBasedMining.MatrixProfile()
window_sizes = [20, 50, 100]
fig, ax = plt.subplots(nrows=1, ncols=3, dpi=150, figsize=(10, 3))
font = {'size': 6}
matplotlib.rc('font', **font)
c = 0
for i in range(len(seqs)):
    for j in range(len(window_sizes)):
        print(i, window_sizes[j])
        f = 'mat/dist_mat_s%d_w%d.np.npy' % (i, window_sizes[j])
        dist_mat = m.distance_matrix(seqs[i], window_sizes[j])
        np.save(f, dist_mat)
        # dist_mat = np.load(f)

        m.plot_distance_matrix(dist_mat, ax=ax[j])
        ax[j].set_title("w=%d" % window_sizes[j])
        c += 1

        for item in ([ax[j].xaxis.label, ax[j].yaxis.label] +
                     ax[j].get_xticklabels() + ax[j].get_yticklabels()):
            item.set_fontsize(8)
fig.tight_layout()
plt.savefig('plots/distance_matrix.png')

fig2, ax2 = plt.subplots(nrows=1, ncols=1, dpi=150, figsize=(8, 2))
for i in range(len(seqs)):
    for j in range(len(window_sizes)):
        print(i, window_sizes[j])
        f = 'mat/dist_mat_s%d_w%d.np.npy' % (i, window_sizes[j])
        dist_mat = np.load(f)

        profile = m.matrix_profile(dist_mat)
        m.plot_matrix_profile(profile, ax=ax2, label='w=%d' % window_sizes[j])
        ax2.set_ylabel('closest subsequence distance')
        if i == 1:
            ax2.set_xlabel('subsequence position')
plt.legend()
fig2.tight_layout()
plt.savefig('plots/matrix_profile.png')

