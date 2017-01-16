# JMVAE

This is a implementation of JMVAE, which is proposed in the following paper:
Masahiro Suzuki, Kotaro Nakayama, Yutaka Matsuo, "Joint Multimodal Learning with Deep Generative Models".

The main code of JMVAE is written on *[Tars](https://github.com/masa-su/Tars) *, which is a framework of deep generation models we are developing. Therefore, please install Tars at first before executing codes in this repository.

# Installation
Please install [Tars](https://github.com/masa-su/Tars)(v0.0.2) as follows.
```
$ git clone https://github.com/masa-su/Tars.git -b v0.0.2
$ pip install -e Tars --process-dependency-links
```
When you execute this command, the following packages will be automatically installed in your environment:

* Theano
* Lasagne
* progressbar2
* matplotlib
* sklearn

# Citation
If you use this code for your researches, please cite our paper
```
@article{suzuki2016joint,
  title={Joint Multimodal Learning with Deep Generative Models},
  author={Suzuki, Masahiro and Nakayama, Kotaro and Matsuo, Yutaka},
  journal={arXiv preprint arXiv:1611.01891},
  year={2016}
}
```
and [Tars](https://github.com/masa-su/Tars).
