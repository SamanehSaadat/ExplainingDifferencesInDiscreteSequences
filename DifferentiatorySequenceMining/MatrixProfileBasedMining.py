import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MatrixProfile:
    def hamming_distance(self, sequence, i, j, w):
        n = len(sequence)
        if i + w > n or j + w > n:
            # Invalid window
            return None
        dist = 0
        for k in range(w):
            if sequence[i + k] != sequence[j + k]:
                dist += 1
        return dist

    def LCS_distance(self, sequence, i, j, w):
        n = len(sequence)
        if i + w > n or j + w > n:
            # Invalid window
            return None
        dist = w - self.LCS(sequence[i: i + w], sequence[j: j + w], w)
        return dist

    def LCS(self, s1, s2, w):
        d = np.zeros((w, w))
        for i in range(1, w):
            for j in range(1, w):
                if s1[i] == s2[i]:
                    d[i, j] = d[i - 1, j - 1] + 1
                else:
                    d[i, j] = max(d[i - 1, j], d[i, j - 1])
        return d[w - 1, w - 1]

    def distance_matrix(self, sequence, window_size, distance_function='hamming'):
        n = len(sequence) - window_size + 1
        dist_mat = np.full((n, n), window_size)
        radius = window_size // 2
        for i in range(n):
            for j in range(i + radius, n):
                dist = np.inf
                if distance_function == 'hamming':
                    dist = self.hamming_distance(sequence, i, j, window_size)
                elif distance_function == 'LCS':
                    dist = self.LCS_distance(sequence, i, j, window_size)
                else:
                    print("ERROR: INVALID DISTANCE FUNCTION")
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
        return dist_mat

    def matrix_profile(self, distance_matrix):
        profile = np.amin(distance_matrix, axis=1)
        return profile

    def plot_distance_matrix(self, distance_matrix, ax):
        im = ax.imshow(distance_matrix, cmap='hot')
        plt.sca(ax)
        plt.colorbar(im, orientation='vertical')
        # plt.show()

    def plot_matrix_profile(self, profile, ax, label):
        pdf = pd.Series(profile)
        pdf.plot(ax=ax, label=label)

    def nearest_neighbor_position_and_distance(self, dist_mat, distance_threshold):
        match = np.argmin(dist_mat, axis=1)
        match_dist = np.amin(dist_mat, axis=1)
        position = pd.Series(dict(zip(range(len(dist_mat)), match)), name='nn_position')
        dist = pd.Series(dict(zip(range(len(dist_mat)), match_dist)), name='nn_dist')
        df = position.to_frame().join(dist).reset_index()
        df = df[df.nn_dist <= distance_threshold].reset_index(drop=True)
        return df

    def get_nearest_neighbor_string(self, sequence, window_size, df):
        df['motif1'] = df['index'].apply(lambda x: self.get_string(sequence, x, window_size))
        df['motif2'] = df['nn_position'].apply(lambda x: self.get_string(sequence, x, window_size))
        return df

    def get_string(self, sequence, start, window_size):
        return sequence[start: start + window_size]
