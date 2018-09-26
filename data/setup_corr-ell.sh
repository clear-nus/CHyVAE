#!/usr/bin/env bash
mkdir corr-ell
wget -c -O corr-ell/correlated_ellipses.npz https://github.com/crslab/correlated-ellipses/raw/master/correlated_ellipses.npz
wget -c -O ../code/synthetic.py https://github.com/crslab/correlated-ellipses/raw/master/synthetic.py
