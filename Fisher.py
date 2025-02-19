import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.decomposition

class PCA:
    def __init__(self, n_components=2, method='svd'):
        self.n_components = n_components
        self.method = method

    def svd(self):
        # Singular value decomposition.
        U, S, Vt = np.linalg.svd(self.X, full_matrices=False)

        # Singular values are already sorted.
        assert np.allclose(S, np.flip(np.sort(S)))

        # X = U * S * Vt
        assert np.allclose(self.X, U * np.diag(S) * Vt)

        # T = U * S
        pcUdS = U[:, :self.n_components] * np.diag(S[:self.n_components])

        # T = X * V
        self.V = Vt.T
        pcXW = self.X * self.V[:, :self.n_components]

        # X * V = U * S
        assert np.allclose(pcXW, pcUdS)

        return pcUdS

    def eig(self):
        # Convariance matrix.
        C = self.X.T * self.X / (self.N - 1)
        assert np.allclose(C, np.cov(self.X.T))

        # Eigendecomposition of the covariance matrix C.
        L, W = np.linalg.eig(C)

        # Sort eigenvectors W in the descending order of eigenvalues L.
        self.W = W[:, np.flip(np.argsort(L))]
        L = np.flip(np.sort(L))
        
        assert np.allclose(C, self.W * np.diag(L) * self.W.T)
        assert np.allclose(self.W * self.W.T, np.identity(len(L)))
        
        # T = X * W
        return self.X * self.W[:, :self.n_components]
        
    def restore(self, X):
        if self.method == 'svd':
            return X * self.V.T + self.mu
        elif self.method == 'eig':
            return X * self.W.T + self.mu
  
    def fit_transform(self, X):
        self.N, self.n = X.shape
        
        # Mean centering.
        self.mu = np.mean(X, axis=0)
        X = X - self.mu
        
        self.X = np.asmatrix(X)
        
        if self.method == 'svd':
            transformed = self.svd()
        elif self.method == 'eig':
            transformed = self.eig()

        return transformed

def get_dataset():
    # Fisher's iris dataset.
    dataset = sklearn.datasets.load_iris()

    y = sklearn.datasets.load_iris()['target']
    X = dataset['data']

    labels = dataset['target_names']
    feature_names = dataset['feature_names']

    # Mean centering.
    mu = np.mean(X, axis=0)
    X = X - mu
    X = np.asmatrix(X)

    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    for i, (label, color) in enumerate(zip(labels, ['red', 'green', 'blue'])):
        plt.scatter(X[y==i, 0].A, X[y==i, 1].A, marker=f'{i+1}', linewidth=0.8, color=color, label=label)
    plt.xlim(-1.8, 2.2)
    plt.ylim(-2, 2)
    plt.axline([0, 0], [1, 0], linestyle='--', linewidth=0.8, color='gray')
    plt.axline([0, 0], [0, 1], linestyle='--', linewidth=0.8, color='gray')
    plt.annotate('$X_1$', xy=(1.95, 0.05))
    plt.annotate('$X_2$', xy=(0.05, 1.8))
    plt.legend()
    plt.savefig('figure/iris_x1_x2.png', dpi=100)

    return X, y, labels

def projection(X, y, labels):
    N, n = X.shape

    # Covariance matrix.
    C = X.T * X / (N - 1)

    assert np.allclose(C, np.cov(X.T))

    print('Covariance matrix\n', C.round(3))

    # Singular value decomposition.
    U, S, Vt = np.linalg.svd(X)

    print('Singular values', S.round(3))
    print('Eigenvalues', np.round(S ** 2 / (N-1), 3))

    print(np.round((S ** 2 / (N-1)) / np.sum(S ** 2 / (N-1)), 3))

    T = np.asmatrix(X) * Vt.T

    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor('white')
    for i, (label, color) in enumerate(zip(labels, ['red', 'green', 'blue'])):
        plt.scatter(T[y==i, 0].A, T[y==i, 1].A, marker=f'{i+1}', linewidth=0.8, color=color, label=label)
    plt.axline([0, 0], [1, 0], linestyle='--', linewidth=0.8, color='gray')
    plt.axline([0, 0], [0, 1], linestyle='--', linewidth=0.8, color='gray')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend()
    plt.savefig('figure/iris_projected.png', dpi=100)

if __name__ == '__main__':
    X, y, labels = get_dataset()

    projection(X, y, labels)

    pca_svd = PCA(n_components=2, method='svd')
    svd_pc = pca_svd.fit_transform(X)

    pca_eig = PCA(n_components=2, method='eig')
    eig_pc = pca_eig.fit_transform(X)

    pca_sklearn = sklearn.decomposition.PCA(n_components=2)
    sklearn_pc = pca_sklearn.fit_transform(np.asarray(X))

    # Projected coodinates are the same except for their signs.
    sklearn_pc[:, 1] = -sklearn_pc[:, 1]

    assert np.allclose(eig_pc, svd_pc)
    assert np.allclose(sklearn_pc, svd_pc)

    pca_svd = PCA(n_components=4, method='svd')
    svd_pc = pca_svd.fit_transform(X)

    # Verify the restored dataset.
    assert np.allclose(X, pca_svd.restore(svd_pc))
