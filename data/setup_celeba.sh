#!/usr/bin/env bash
mkdir celeba
echo 'Downloading dataset...'
python download_celeba.py
echo 'Preprocessing...'
python preprocess_celeba.py
rm -rf tmp/