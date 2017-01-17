import numpy as np
import theano
import time
import pickle
import tempfile
import shutil

from Tars.model.mvae_old import MVAE_OLD
from Tars.distribution import Bernoulli, Gaussian, GaussianConstantVar, Categorical
from Tars.load_data import mnist, celeba, flickr

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, batch_norm, FlattenLayer, ReshapeLayer, Deconv2DLayer, ConcatLayer
from lasagne.nonlinearities import sigmoid, tanh, softplus, rectify, elu, linear, softmax, leaky_rectify, identity
from lasagne.updates import sgd, momentum, adagrad, adadelta, rmsprop, adam

import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from progressbar import ProgressBar

DATAPATH = os.getenv("HOME") + "/share/data/"

from train_jmvae_kl_z_x import plot_x, bernoullisample


def train(data, activation, plot_image, rseed, n_epoch, Optimizer, l, k, sample_l, sample_k, sampling_type, n_batch, annealing, annealing_epoch, bn_layer, options_dict):
    np.random.seed(rseed)
    rng = np.random.RandomState(rseed)

    if bn_layer is True:
        bn = batch_norm
    else:
        bn = lambda x:x

    if data == "mnist":
        load, plot = mnist(DATAPATH)
        train_x, train_y, valid_x, valid_y, test_x, test_y = load(test=True)
        train_x = np.concatenate([train_x,valid_x])
        train_y = np.concatenate([train_y,valid_y])
        test_x = bernoullisample(test_x, rng)

        size = (28, 28)
        n_x = (28 * 28)
        n_z = 64
        n_y = 10

        x0 = InputLayer((None, n_x))
        x1 = InputLayer((None, n_y))

        q0_0 = bn(DenseLayer(
            x0, num_units=512, nonlinearity=activation))
        q1_0 = bn(DenseLayer(
            x1, num_units=512, nonlinearity=activation))
        q_1 = bn(DenseLayer(ConcatLayer([q0_0, q1_0]),num_units=512,nonlinearity=activation))
        q_mean = DenseLayer(q_1, num_units=n_z, nonlinearity=linear)
        q_var = DenseLayer(q_1, num_units=n_z, nonlinearity=softplus)
        q = Gaussian(q_mean, q_var, given=[x0, x1])

        z = InputLayer((None, n_z))
        p0_0 = bn(DenseLayer(
            z, num_units=512, nonlinearity=activation))
        p0_1 = bn(DenseLayer(
            p0_0, num_units=512, nonlinearity=activation))
        p0_mean = DenseLayer(p0_1, num_units=n_x, nonlinearity=sigmoid)
        p0 = Bernoulli(p0_mean, given=[z])

        p1_0 = bn(DenseLayer(
            z, num_units=512, nonlinearity=activation))
        p1_1 = bn(DenseLayer(
            p1_0, num_units=512, nonlinearity=activation))
        p1_mean = DenseLayer(p1_1, num_units=n_y, nonlinearity=softmax)
        p1 = Categorical(p1_mean, given=[z])

        p = [p0, p1]

    elif data == "mnist_2dim":
        load, plot = mnist(DATAPATH)
        train_x, train_y, valid_x, valid_y, test_x, test_y = load(test=True)
        train_x = np.concatenate([train_x,valid_x])
        train_y = np.concatenate([train_y,valid_y])
        test_x = bernoullisample(test_x, rng)

        size = (28, 28)
        n_x = (28 * 28)
        n_z = 2
        n_y = 10

        x0 = InputLayer((None, n_x))
        x1 = InputLayer((None, n_y))

        q0_0 = bn(DenseLayer(
            x0, num_units=512, nonlinearity=activation))
        q1_0 = bn(DenseLayer(
            x1, num_units=512, nonlinearity=activation))
        q_1 = bn(DenseLayer(ConcatLayer([q0_0, q1_0]),num_units=512,nonlinearity=activation))
        q_mean = DenseLayer(q_1, num_units=n_z, nonlinearity=linear)
        q_var = DenseLayer(q_1, num_units=n_z, nonlinearity=softplus)
        q = Gaussian(q_mean, q_var, given=[x0, x1])

        z = InputLayer((None, n_z))
        p0_0 = bn(DenseLayer(
            z, num_units=512, nonlinearity=activation))
        p0_1 = bn(DenseLayer(
            p0_0, num_units=512, nonlinearity=activation))
        p0_mean = DenseLayer(p0_1, num_units=n_x, nonlinearity=sigmoid)
        p0 = Bernoulli(p0_mean, given=[z])

        p1_0 = bn(DenseLayer(
            z, num_units=512, nonlinearity=activation))
        p1_1 = bn(DenseLayer(
            p1_0, num_units=512, nonlinearity=activation))
        p1_mean = DenseLayer(p1_1, num_units=n_y, nonlinearity=softmax)
        p1 = Categorical(p1_mean, given=[z])

        p = [p0, p1]

    else:
        sys.exit()

    model = MVAE_OLD(q, p, n_batch, Optimizer, l=l, random=rseed)

    pbar = ProgressBar(maxval=n_epoch).start()
    lowerbound_all = []
    n_sample = 100
    sample_z = np.random.standard_normal((n_batch, n_z)).astype(np.float32)

    t = int(time.time())
    output_dir = 'plot/%d' % t

    dirpath = tempfile.mkdtemp()
    f = open(os.path.join(dirpath,"paramaters.txt"), "w")
    for key in options_dict.keys():
        f.write("#%s=%s\n" % (key, str(options_dict[key])))
    f.write("#file=%s\n" % __file__)
    f.close()
    shutil.copy(os.path.realpath(__file__), os.path.join(dirpath, __file__))

    if annealing is True:
        annealing_beta = 0
    else:
        annealing_beta = 1
        annealing_epoch = 1

    for i in range(1, n_epoch + 1):
        train_x, train_y = shuffle(train_x, train_y)
        lowerbound_train = model.train([bernoullisample(train_x, rng), train_y], annealing_beta=annealing_beta)
        if i < annealing_epoch:
            annealing_beta = i / float(annealing_epoch - 1)
        if ((i % 100 == 0) or (i == 1)) and (plot_image is True):
            log_likelihood_test = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l, mode=sampling_type, n_batch=10)
            log_conditional_likelihood_test = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l, mode=sampling_type, type_p="conditional", n_batch=10, sampling_n=5000)
            log_mg_likelihood_test = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l, mode=sampling_type, type_p="marginal", n_batch=10)
            log_pseudo_mg_likelihood_test = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l, mode=sampling_type, type_p="pseudo_marginal", n_batch=10)
            log_pseudo_conditional_likelihood_test = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l, mode=sampling_type, type_p="pseudo_conditional", n_batch=10, sampling_n=5000)

            lw = "epoch = %d lower bound (train) = %lf ( %lf %lf %lf ) log likelihood (test) = %lf conditional log likelihood (test) = %lf mg log likelihood (test) = %lf pseudo_mg log likelihood (test) = %lf pseudo_conditional log likelihood (test) = %lf\n" % (
                i, sum(lowerbound_train), lowerbound_train[0], lowerbound_train[1], lowerbound_train[2], np.mean(log_likelihood_test), np.mean(log_conditional_likelihood_test), np.mean(log_mg_likelihood_test), np.mean(log_pseudo_mg_likelihood_test), np.mean(log_pseudo_conditional_likelihood_test))

            f = open(os.path.join(dirpath, "temp.txt"), "a")
            f.write(lw)
            f.close()
            print lw[:-1]

            try:
                model_path = os.path.join(dirpath, "p.pkl")
                with open(model_path, "w") as f:
                    pickle.dump(p, f)
                print "save %s" % model_path
            except:
                sys.exit()

            try:
                model_path = os.path.join(dirpath, "q.pkl")
                with open(model_path, "w") as f:
                    pickle.dump(q, f)
                print "save %s" % model_path
            except:
                sys.exit()

            plot_x(model,plot,sample_z,i,n_sample,dirpath)
        
        pbar.update(i)

    os.rename(dirpath, output_dir)
    return lowerbound_all

if __name__ == "__main__":

    from optparse import OptionParser

    usage = u'%prog [Options]\nDetailed options -h or --help'
    parser = OptionParser(usage=usage)

    parser.add_option(
        '--dataset',
        action='store',
        type='str',
        dest='data',
        help='Set dataset (ex. mnist, svhn)'
    )
    parser.set_defaults(data='mnist')

    parser.add_option(
        '--activation',
        action='store',
        type='str',
        dest='activation',
        help='Set activation function (ex. relu)'
    )
    parser.set_defaults(activation='relu')

    parser.add_option(
        '--plot',
        action='store_true',
        dest='plot_image',
        help='Set plot image flag'
    )
    parser.set_defaults(plot_image=True)

    parser.add_option(
        '--rseed',
        action='store',
        type='int',
        dest='rseed',
        help='Set random seed'
    )
    parser.set_defaults(rseed=1)

    parser.add_option(
        '--epoch',
        action='store',
        type='int',
        dest='n_epoch',
        help='Set the number of epoch'
    )
    parser.set_defaults(n_epoch=100)

    parser.add_option(
        '--optimizer',
        action='store',
        type='str',
        dest='optimizer',
        help='Set optimizer (ex. Adam)'
    )
    parser.set_defaults(optimizer='Adam')

    parser.add_option(
        '--l',
        action='store',
        type='int',
        dest='l',
        help='Set the nunber of sampling l'
    )
    parser.set_defaults(l=1)

    parser.add_option(
        '--k',
        action='store',
        type='int',
        dest='k',
        help='Set the nunber of sampling k'
    )
    parser.set_defaults(k=1)

    parser.add_option(
        '--sample_l',
        action='store',
        type='int',
        dest='sample_l',
        help='Set the nunber of sampling sample l'
    )
    parser.set_defaults(sample_l=1000)

    parser.add_option(
        '--sample_k',
        action='store',
        type='int',
        dest='sample_k',
        help='Set the nunber of sampling sample k'
    )
    parser.set_defaults(sample_k=1000)

    parser.add_option(
        '--sampling_type',
        action='store',
        dest='sampling_type',
        help='set importance weight sampling type',
        choices=["iw", "lowerbound"]
    )
    parser.set_defaults(sampling_type="iw")

    parser.add_option(
        '--nbatch',
        action='store',
        type='int',
        dest='n_batch',
        help='Set the number of batch'
    )
    parser.set_defaults(n_batch=100)

    parser.add_option(
        '--annealing',
        action='store_true',
        dest='annealing',
        help='Set annealing'
    )
    parser.set_defaults(annealing=False)

    parser.add_option(
        '--annealing_epoch',
        action='store',
        type='int',
        dest='annealing_epoch',
        help='Set the number of annealing epoch'
    )
    parser.set_defaults(annealing_epoch=100)

    parser.add_option(
        '--bn',
        action='store_true',
        dest='bn_layer',
        help='Set whether batch_norm'
    )
    parser.set_defaults(bn_layer=False)

    activations = {'sigmoid': sigmoid,
                   'tanh': tanh,
                   'softplus': softplus,
                   'relu': rectify,
                   'elu': elu,
                   'leaky_relu': leaky_rectify}

    optimizers = {'SGD': sgd,
                  'MomentumSGD': momentum,
                  'RMSprop': rmsprop,
                  'AdaDelta': adadelta,
                  'AdaGrad': adagrad,
                  'Adam': adam}

    options, args = parser.parse_args()
    options_dict = vars(options)

    for key in options_dict.keys():
        print "#%s=%s" % (key, str(options_dict[key]))

    lowerbound_all = train(options.data,
                           activations[options.activation],
                           options.plot_image,
                           options.rseed,
                           options.n_epoch,
                           optimizers[options.optimizer],
                           options.l,
                           options.k,
                           options.sample_l,
                           options.sample_k,
                           options.sampling_type,
                           options.n_batch,
                           options.annealing,
                           options.annealing_epoch,
                           options.bn_layer,
                           options_dict)
