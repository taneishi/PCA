import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import sklearn.datasets

def scale(df, center=True, scale=True):
    # compatible with R scale()
    if center:
        df = df - df.mean()
    if scale:
        df = df / df.std()
    return df

class PCA:
    def __init__(self, df, npc=2, method='svd'):
        self.df = df
        self.npc = npc
        self.method = method

    def svd(self):
        self.df = self.df - np.mean(self.df, axis=0)

        # SVD decomposite X to U*diag(S)*Wt
        U, diagS, Wt = np.linalg.svd(self.df, full_matrices=False)

        # singular values are already sorted
        assert np.all(diagS[:-1] >= diagS[1:])

        # T = X*W = U*S
        pcXW = np.dot(self.df, Wt.T[:, :self.npc])
        pcUdS = U[:, :self.npc] * diagS[:self.npc]

        # these results should be the same, but there are a little differences.
        assert np.allclose(pcXW, pcUdS, atol=1e-5)

        return pcUdS

    def eig(self):
        # calculate eigenvalues(l) and eigenvectors(w) of the covariance matrix
        C = np.cov(self.df.T)
        l, w = np.linalg.eig(C)

        # sort eigenvectors by eigenvalue in descending order
        w = w[:, np.argsort(l)[::-1]]

        # T = X*W
        return np.dot(self.df, w[:, :self.npc])

    def pc(self):
        if self.method == 'svd':
            score = self.svd()
        else:
            score = self.eig()

        # flip
        score[:, 1] = -score[:, 1]

        return score

def plot(df, score, columns):
    df = pd.concat([df, score], axis=1)
    for name in df['Species'].unique():
        cond = df['Species'] == name
        plt.plot(df.loc[cond, columns[0]], df.loc[cond, columns[1]], 'o', label=name)
    plt.grid(True)
    plt.legend(framealpha=0.5)

def main(columns=['PC1', 'PC2']):
    np.set_printoptions(precision=4, threshold=30)

    # Fisher iris data
    data = sklearn.datasets.load_iris()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['Species'] = [data['target_names'][target] for target in data['target']]

    # select only numerical columns
    values = scale(df.iloc[:, :4], center=True, scale=True)
    print('DataFrame is a %dx%d matrix of iris.csv' % (values.shape))

    scores = {}

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Singular value decomposition')
    pca_svd = PCA(values, method='svd')
    score = pd.DataFrame(pca_svd.pc(), columns=columns)
    plot(df, score, columns)
    scores['svd'] = score

    plt.subplot(1, 3, 2)
    plt.title('Eigenvalue decomposition')
    pca_eig = PCA(values, method='eig')
    score = pd.DataFrame(pca_eig.pc(), columns=columns)
    plot(df, score, columns)
    scores['eig'] = score

    plt.subplot(1, 3, 3)
    plt.title('Principal Component Analysis of scikit-learn')
    pca_sklearn = sklearn.decomposition.PCA(n_components=2)
    score = pca_sklearn.fit_transform(values)
    score = pd.DataFrame(score, columns=columns)
    plot(df, score, columns)
    scores['sklearn'] = score

    plt.tight_layout()
    plt.savefig('figure/pca.png')

    # assert if the three methods match
    print('PCA(svd) == PCA(eig) is %s' % (np.allclose(scores['svd'], scores['eig'])))
    print('PCA(svd) == PCA(sklearn) is %s' % (np.allclose(scores['svd'], scores['sklearn'])))

if __name__ == '__main__':
    main()
