# JMVAE


Code for reproducing results of our paper *["Joint Multimodal Learning with Deep Generative Models"](https://arxiv.org/abs/1611.01891)*

We also developed a Python framework for deep generative models called *[Tars](https://github.com/masa-su/Tars)* in Theano and Lasagne. As we implemented JMVAE as a model in Tars, please install this framework first before executing codes in this repository.

# Installation
Run the following comamnds to install [Tars](https://github.com/masa-su/Tars) (v0.0.2).
Please make sure that you specify the version 0.0.2.
```
$ git clone https://github.com/masa-su/Tars.git -b v0.0.2
$ pip install -e Tars --process-dependency-links
```
When you execute the above commands, the following packages will be automatically installed in your environment:

* Theano
* Lasagne
* progressbar2
* matplotlib
* sklearn

# Training models
Use ```main_jmvae_zero_z_x.py``` and ```main_jmvae_kl_z_x.py``` scripts to train JMVAE-zero and JMVAE-kl on the MNIST.
If you want to train JMVAE-GAN on CelebA, first download [the CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Then crop the images to 64×64, and put them to ```Tars/datasets``` directory before executing ```main_jmvaegan_kl_z_x.py```.

# Citation
If you use this code for your research, please cite our paper:
```
@article{suzuki2016joint,
  title={Joint Multimodal Learning with Deep Generative Models},
  author={Suzuki, Masahiro and Nakayama, Kotaro and Matsuo, Yutaka},
  journal={arXiv preprint arXiv:1611.01891},
  year={2016}
}
```
and [Tars](https://github.com/masa-su/Tars).
