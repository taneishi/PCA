import numpy as np
import sklearn.datasets

# For verificatoin.
import sklearn.decomposition

from models import PCA

def main(n_components=2):
    # Fisher iris dataset.
    data = sklearn.datasets.load_iris()
    X = data['data']

    pca = PCA(n_components=n_components, method='eig')
    eig_pc = pca.fit_transform(X)
    
    pca = PCA(n_components=n_components, method='svd')
    svd_pc = pca.fit_transform(X)

    pca = sklearn.decomposition.PCA(n_components=n_components)
    sklearn_pc = pca.fit_transform(X)

    # Principal components are the same except for the signs.
    sklearn_pc[:, 1] = -sklearn_pc[:, 1]

    # Assert that SVD and Eig results the same.
    assert np.allclose(eig_pc, svd_pc)
    assert np.allclose(sklearn_pc, svd_pc)

    pca = PCA(n_components=X.shape[1], method='svd')
    svd_pc = pca.fit_transform(X)

    # Verify the restoration.
    assert np.allclose(X, pca.restore(svd_pc))

if __name__ == '__main__':
    main()
