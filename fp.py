import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
from sklearn import preprocessing
import sklearn.decomposition
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import timeit

from pca import PCA

def fp_mds(args, radius, method):
    fps = []
    bit_info = {}
    solubility = []
    for mol in Chem.SDMolSupplier(args.filename):
        fp_bit = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, bitInfo=bit_info, nBits=args.nbits))

        # you can use fp_bit for fixed length fingerprints,
        # while this code generates matrix from variable length fingerprints for generality.
        fp = list(bit_info.keys())
        fps.append(fp)
        solubility.append(np.float32(mol.GetProp('SOL')))

    mat = np.zeros((len(fps), args.nbits), dtype=np.float32)
    for i, fp in enumerate(fps):
        mat[i, fp] = 1.

    mat = preprocessing.normalize(mat)

    start_time = timeit.default_timer()

    if method == 'eig':
        pcs = np.real(PCA(mat, npc=2, method='eig').pc())
    elif method == 'full':
        pcs = sklearn.decomposition.PCA(n_components=2, svd_solver='full').fit_transform(mat)
    elif method == 'randomized':
        pcs = sklearn.decomposition.PCA(n_components=2, svd_solver='randomized').fit_transform(mat)
    print('radius %2d matrix shape %s %5.2f sec' % (radius, mat.shape, timeit.default_timer() - start_time))

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(pcs[:, 0].min(), pcs[:, 0].max()), np.linspace(pcs[:, 1].min(), pcs[:, 1].max())
    xi, yi = np.meshgrid(xi, yi)

    # Interpolation
    rbf = scipy.interpolate.Rbf(pcs[:, 0], pcs[:, 1], solubility, function='linear', smooth=0.1)
    zi = rbf(xi, yi)

    return zi, pcs, solubility

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename', default='data/solubility.test.sdf')
    parser.add_argument('--nbits', default=4096, type=int)
    args = parser.parse_args()
    print(vars(args))

    for method in ['eig', 'full', 'randomized']:
        if method == 'eig':
            print('PCA using implemented in pca.py')
        elif method == 'full':
            print('PCA using sklearn full SVD')
        elif method == 'randomized':
            print('PCA using sklearn truncated SVD')
        plt.figure(figsize=(9, 6))
        for index, radius in enumerate(range(0, 11, 2), 1):
            zi, pcs, solubility = fp_mds(args, radius, method)

            plt.subplot(2, 3, index)
            plt.title('ecfp%d %dbits' % (radius, args.nbits))
            plt.imshow(zi, vmin=zi.min(), vmax=zi.max(), origin='lower', cmap='RdYlGn_r', aspect='auto',
                    extent=[pcs[:, 0].min(), pcs[:, 0].max(), pcs[:, 1].min(), pcs[:, 1].max()])
            plt.scatter(pcs[:, 0], pcs[:, 1], c=solubility, cmap='RdYlGn_r')

        plt.tight_layout()
        plt.savefig('figure/fp_%s.png' % (method))

if __name__ == '__main__':
    main()
