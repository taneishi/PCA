#!/bin/bash

mkdir -p figure data

wget -c -P data https://github.com/rdkit/rdkit/raw/master/Docs/Book/data/solubility.test.sdf

python pca.py
python fp.py
