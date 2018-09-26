# CHyVAE
Code for our paper [Hyperprior Induced Unsupervised Disentanglement of Latent Representations](https://arxiv.org/abs/1809.04497) (AAAI-19)

## Requirements

* Python 3
* Tensorflow (tested on 1.10.1)
* Numpy (tested on 1.14.5)
* OpenCV (tested on 3.4.3)


## Usage

### Setting up the datasets
Traverse to `data/` and run `setup_2dshapes.sh` and `setup_corr-ell.sh` to set up `2dshapes` and `correlated_ellipses` datasets.

### Training a model

Traverse to `code/` and run
```
python main.py \
       --dataset [2dshapes/correlated_ellipses] \
       --z_dim [dim. of latent space] \
       --n_steps [number of training steps] \
       --nu [degrees of freedom] \
       --batch_size [batch size]
```
The reconstruction error and disentanglement metric will be logged at a set interval as training proceeds.

**Example Run**
```
python main.py --dataset correlated_ellipses --z_dim 10 --n_steps 150000 --nu 200 --batch_size 50
```

Run `python main.py -h` for help.

## Datasets

Currently the repository includes code for experimenting on the following datasets.

* 2DShapes
* CorrelatedEllipses

## Additional Results
For additonal qualitative results, please check [AdditionalResults.md](AdditionalResults.md)

## Contact
For any questions regarding the code or the paper, please email abdulfatir@u.nus.edu

## BibTeX

```
@inproceedings{ansari2018hyperprior,
  title={Hyperprior Induced Unsupervised Disentanglement of Latent Representations},
  author={Ansari, Abdul Fatir and Soh, Harold},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```
