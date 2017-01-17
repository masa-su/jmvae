import sys
import os
import numpy as np
import time
import pickle
import tempfile
import matplotlib
import shutil

from lasagne.layers import (
    InputLayer,
    DenseLayer,
    Conv2DLayer,
    batch_norm,
    FlattenLayer,
    ReshapeLayer,
    Deconv2DLayer,
    ConcatLayer,
)
from lasagne.nonlinearities import (
    sigmoid,
    tanh,
    softplus,
    rectify,
    elu,
    linear,
    leaky_rectify,
    softmax,
)
from lasagne.updates import sgd, momentum, adagrad, adadelta, rmsprop, adam

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from progressbar import ProgressBar
from collections import OrderedDict

from Tars.model import MVAEGAN
from Tars.distribution import (
    Bernoulli,
    Gaussian,
    Categorical,
    GaussianConstantVar,
)
from Tars.load_data import mnist, celeba

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.setrecursionlimit(5000)

DATAPATH = os.getenv("HOME") + "/share/data/"


def plot_sample(plot, sample_x, path):
    fig = plt.figure(figsize=(10, 10))
    X, cmap = plot(sample_x)

    for j, x in enumerate(X):
        ax = fig.add_subplot(10, 10, j + 1)
        ax.imshow(x, cmap)
        ax.axis('off')

    plt.savefig('%s/sample.jpg' % path)
    plt.close()


def plot_single_x(model, plot, sample_x, i, path):
    sample_z = model.pq0_sample_mean_x(sample_x)
    reconst_x = model.p0_sample_mean_x(sample_z)
    fig = plt.figure(figsize=(10, 10))
    X, cmap = plot(reconst_x)

    for j, x in enumerate(X):
        ax = fig.add_subplot(10, 10, j + 1)
        ax.imshow(x, cmap)
        ax.axis('off')

    plt.savefig('%s/single_%04d.jpg' % (path, i))
    plt.close()


def plot_multiple_x(model, plot, sample_x, sample_y, i, path):
    sample_z = model.q_sample_mean_x(sample_x, sample_y)
    reconst_x = model.p0_sample_mean_x(sample_z)
    fig = plt.figure(figsize=(10, 10))
    X, cmap = plot(reconst_x)

    for j, x in enumerate(X):
        ax = fig.add_subplot(10, 10, j + 1)
        ax.imshow(x, cmap)
        ax.axis('off')

    plt.savefig('%s/multi_%04d.jpg' % (path, i))
    plt.close()


def train(data, activation, plot_image, rseed, n_epoch, Optimizer,
          l, k, sample_l, sample_k, gamma, gan_gamma,
          sampling_type, n_batch,
          annealing, annealing_epoch, bn_layer, options_dict):

    np.random.seed(rseed)
    rng = np.random.RandomState(rseed)

    if bn_layer is True:
        bn = batch_norm
    else:
        def bn(x): return x

    if data == "mnist":
        load, plot = mnist(DATAPATH)
        train_x, train_y, _, _, test_x, test_y = load(test=True)

        n_x = (28 * 28)
        n_z = 64
        n_y = 10

        x0 = InputLayer((None, n_x), name="x0")
        x1 = InputLayer((None, n_y), name="x1")

        q0_0 = DenseLayer(x0, num_units=512, nonlinearity=activation)
        q0_1 = DenseLayer(q0_0, num_units=512, nonlinearity=activation)
        q1_0 = DenseLayer(x1, num_units=512, nonlinearity=activation)
        q1_1 = DenseLayer(q1_0, num_units=512, nonlinearity=activation)
        q_1 = ConcatLayer([q0_1, q1_1])
        q_mean = DenseLayer(q_1, num_units=n_z, nonlinearity=linear)
        q_var = DenseLayer(q_1, num_units=n_z, nonlinearity=softplus)
        q = Gaussian(q_mean, q_var, given=[x0, x1])

        z = InputLayer((None, n_z), name="z")
        p0_0 = DenseLayer(z, num_units=512, nonlinearity=activation)
        p0_1 = DenseLayer(p0_0, num_units=512, nonlinearity=activation)
        p0_mean = DenseLayer(p0_1, num_units=n_x, nonlinearity=sigmoid)
        p0 = Bernoulli(p0_mean, given=[z])

        p1_0 = DenseLayer(z, num_units=512, nonlinearity=activation)
        p1_1 = DenseLayer(p1_0, num_units=512, nonlinearity=activation)
        p1_mean = DenseLayer(p1_1, num_units=n_y, nonlinearity=softmax)
        p1 = Categorical(p1_mean, given=[z])

        p = [p0, p1]

        q0_0 = DenseLayer(x0, num_units=512, nonlinearity=activation)
        q0_1 = DenseLayer(q0_0, num_units=512, nonlinearity=activation)
        q0_mean = DenseLayer(q0_1, num_units=n_z, nonlinearity=linear)
        q0_var = DenseLayer(q0_1, num_units=n_z, nonlinearity=softplus)
        q0 = Gaussian(q0_mean, q0_var, given=[x0])

        q1_0 = DenseLayer(x1, num_units=512, nonlinearity=activation)
        q1_1 = DenseLayer(q1_0, num_units=512, nonlinearity=activation)
        q1_mean = DenseLayer(q1_1, num_units=n_z, nonlinearity=linear)
        q1_var = DenseLayer(q1_1, num_units=n_z, nonlinearity=softplus)
        q1 = Gaussian(q1_mean, q1_var, given=[x1])

        pq = [q0, q1]

        d_0 = DenseLayer(x0, num_units=512, nonlinearity=leaky_rectify)
        d_1 = DenseLayer(d_0, num_units=512, nonlinearity=leaky_rectify)
        d_mean = DenseLayer(d_1, num_units=1, nonlinearity=sigmoid)
        d = Bernoulli(d_mean, given=[x0])

    elif data == "celeba":
        load, plot, preprocess = celeba(DATAPATH, toFloat=True)
        x, y = load()

        n_x = (64 * 64)
        n_z = 128

        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=0.01, random_state=rseed)

        n_y = len(train_y[0])

        x0 = InputLayer((None, 3, 64, 64))
        x1 = InputLayer((None, n_y))
        z = InputLayer((None, n_z))

        q0_0 = Conv2DLayer(x0, num_filters=64, filter_size=4,
                           stride=2, pad=1, nonlinearity=activation)
        q0_1 = batch_norm(Conv2DLayer(q0_0, num_filters=64 * 2,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q0_2 = batch_norm(Conv2DLayer(q0_1, num_filters=64 * 4,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q0_3 = batch_norm(Conv2DLayer(q0_2, num_filters=64 * 4,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q1_0 = DenseLayer(x1, num_units=512, nonlinearity=activation)
        q1_1 = batch_norm(DenseLayer(
            q1_0, num_units=512, nonlinearity=activation))
        q_1 = DenseLayer(ConcatLayer(
            [FlattenLayer(q0_3), q1_1]), num_units=1024,
                                    nonlinearity=activation)
        q_mean = DenseLayer(q_1, num_units=n_z, nonlinearity=linear)
        q_var = DenseLayer(q_1, num_units=n_z, nonlinearity=softplus)
        q = Gaussian(q_mean, q_var, given=[x0, x1])

        p0_0 = DenseLayer(z, num_units=64 * 4 * 4 * 4, nonlinearity=activation)
        p0_1 = ReshapeLayer(p0_0, (([0], 64 * 4, 4, 4)))
        p0_2 = batch_norm(Deconv2DLayer(p0_1, num_filters=64 * 4,
                                        filter_size=4, stride=2, crop=1,
                                        nonlinearity=activation))
        p0_3 = batch_norm(Deconv2DLayer(p0_2, num_filters=64 * 2,
                                        filter_size=4, stride=2, crop=1,
                                        nonlinearity=activation))
        p0_4 = batch_norm(Deconv2DLayer(
            p0_3, num_filters=64, filter_size=4, stride=2, crop=1,
            nonlinearity=activation))
        p0_mean = Deconv2DLayer(
            p0_4, num_filters=3, filter_size=4, stride=2, crop=1,
            nonlinearity=linear)
        p0 = GaussianConstantVar(p0_mean, given=[z])

        p1_0 = DenseLayer(z, num_units=64 * 4 * 4 * 4, nonlinearity=activation)
        p1_1 = batch_norm(DenseLayer(
            p1_0, num_units=512, nonlinearity=activation))
        p1_mean = DenseLayer(p1_1, num_units=n_y, nonlinearity=tanh)
        p1 = GaussianConstantVar(p1_mean, given=[z])

        p = [p0, p1]

        q0_0 = Conv2DLayer(x0, num_filters=64, filter_size=4,
                           stride=2, pad=1, nonlinearity=activation)
        q0_1 = batch_norm(Conv2DLayer(q0_0, num_filters=64 * 2,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q0_2 = batch_norm(Conv2DLayer(q0_1, num_filters=64 * 4,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q0_3 = batch_norm(Conv2DLayer(q0_2, num_filters=64 * 4,
                                      filter_size=4, stride=2, pad=1,
                                      nonlinearity=activation))
        q0_4 = DenseLayer(FlattenLayer(q0_3), num_units=1024,
                          nonlinearity=activation)
        q0_mean = DenseLayer(q0_4, num_units=n_z, nonlinearity=linear)
        q0_var = DenseLayer(q0_4, num_units=n_z, nonlinearity=softplus)
        q0 = Gaussian(q0_mean, q0_var, given=[x0])

        q1_0 = DenseLayer(x1, num_units=512, nonlinearity=activation)
        q1_1 = batch_norm(DenseLayer(
            q1_0, num_units=512, nonlinearity=activation))
        q1_mean = DenseLayer(q1_1, num_units=n_z, nonlinearity=linear)
        q1_var = DenseLayer(q1_1, num_units=n_z, nonlinearity=softplus)
        q1 = Gaussian(q1_mean, q1_var, given=[x1])

        pq = [q0, q1]


        d_0 = Conv2DLayer(x0, num_filters=64, filter_size=4,
                          stride=2, pad=1, nonlinearity=leaky_rectify)
        d_1 = Conv2DLayer(d_0, num_filters=64 * 2, filter_size=4,
                          stride=2, pad=1, nonlinearity=leaky_rectify)
        d_2 = Conv2DLayer(d_1, num_filters=64 * 4, filter_size=4,
                          stride=2, pad=1, nonlinearity=leaky_rectify)
        d_3 = Conv2DLayer(d_2, num_filters=64 * 4, filter_size=4,
                          stride=2, pad=1, nonlinearity=leaky_rectify)
        d_4 = DenseLayer(FlattenLayer(d_3), num_units=1024,
                         nonlinearity=leaky_rectify)
        d_mean = DenseLayer(d_4, num_units=1, nonlinearity=sigmoid)
        d = Bernoulli(d_mean, given=[x0])

    else:
        sys.exit()

    model = MVAEGAN(q, p, pq, d, n_batch, Optimizer, l=l,
                    random=rseed, gamma=gamma, gan_gamma=gan_gamma)

    pbar = ProgressBar(maxval=n_epoch).start()
    n_sample = 100
    choice = np.random.choice(len(test_x), n_sample)
    sample_x = test_x[choice]
    sample_y = test_y[choice]

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

    plot_sample(plot, sample_x, dirpath)

    for i in range(1, n_epoch + 1):
        shuffle(train_x, train_y)
        lowerbound_train = model.train([train_x, train_y], n_z, rng,
                                       annealing_beta=annealing_beta)

        if i < annealing_epoch:
            annealing_beta = i / float(annealing_epoch - 1)

        if ((i % 10 == 0) or (i == 1)) and (plot_image is True):
            lw = OrderedDict()
            lw["epoch"] = i
            lw["lower bound (train)"] = sum(lowerbound_train[:3])
            lw["lower bound (train) [0]"] = lowerbound_train[0]
            lw["lower bound (train) [1]"] = lowerbound_train[1]
            lw["lower bound (train) [2]"] = lowerbound_train[2]
            
            lw["log likelihood (test) "] = model.log_likelihood_test(
                [test_x, test_y], k=sample_k, l=sample_l,
                mode=sampling_type).mean()
            lw["conditional log likelihood (test) "] \
                = model.log_likelihood_test(
                    [test_x, test_y], k=sample_k, l=sample_l,
                    mode=sampling_type, sampling_n=10,
                    type_p="conditional").mean()
            lw["marginal log likelihood (test) "] \
                = model.log_likelihood_test(
                    [test_x, test_y], k=sample_k, l=sample_l,
                    mode=sampling_type,
                    type_p="marginal").mean()
            lw["pseudo marginal log likelihood (test) "] \
                = model.log_likelihood_test(
                    [test_x], k=sample_k, l=sample_l,
                    mode=sampling_type,
                    type_p="pseudo_marginal").mean()
            lw["pseudo conditional log likelihood (test) "] \
                = model.log_likelihood_test(
                    [test_x, test_y], k=sample_k, l=sample_l,
                    mode=sampling_type, sampling_n=10,
                    type_p="pseudo_conditional").mean()

            lowerbound_test = model.gan_test([test_x, test_y], n_z, rng)
            lw["loss (train) [0]"] = lowerbound_train[5]
            lw["loss (train) [1]"] = lowerbound_train[6]
            lw["loss (test) [0]"] = lowerbound_test[0]
            lw["loss (test) [1]"] = lowerbound_test[1]

            f = open(os.path.join(dirpath, "temp.txt"), "a")
            for key, val in lw.items():
                f.write("%s = %s " % (key, str(val)))
                print "%s = %s" % (key, str(val))
            f.close()

            with open(os.path.join(dirpath, "temp.dump"), "w") as f:
                pickle.dump(lw, f)

            model_path = os.path.join(dirpath, "p.pkl")
            with open(model_path, "w") as f:
                pickle.dump(p, f)
            print "save %s" % model_path

            model_path = os.path.join(dirpath, "q.pkl")
            with open(model_path, "w") as f:
                pickle.dump(q, f)
            print "save %s" % model_path

            model_path = os.path.join(dirpath, "d.pkl")
            with open(model_path, "w") as f:
                pickle.dump(d, f)
            print "save %s" % model_path

            model_path = os.path.join(dirpath, "pq.pkl")
            with open(model_path, "w") as f:
                pickle.dump(pq, f)
            print "save %s" % model_path

            plot_single_x(model, plot, sample_x, i, dirpath)
            plot_multiple_x(model, plot, sample_x, sample_y, i, dirpath)

        pbar.update(i)

    os.rename(dirpath, output_dir)

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
    parser.set_defaults(data='celeba')

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
    parser.set_defaults(sample_l=10)

    parser.add_option(
        '--sample_k',
        action='store',
        type='int',
        dest='sample_k',
        help='Set the nunber of sampling sample k'
    )
    parser.set_defaults(sample_k=10)

    parser.add_option(
        '--gan_gamma',
        action='store',
        type='float',
        dest='gan_gamma',
        help='gan paramater'
    )
    parser.set_defaults(gan_gamma=1)

    parser.add_option(
        '--gamma',
        action='store',
        type='float',
        dest='gamma',
        help='mvae paramater'
    )
    parser.set_defaults(gamma=0.1)

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
    parser.set_defaults(annealing=True)

    parser.add_option(
        '--annealing_epoch',
        action='store',
        type='int',
        dest='annealing_epoch',
        help='Set the number of annealing epoch'
    )
    parser.set_defaults(annealing_epoch=20)

    parser.add_option(
        '--bn',
        action='store_true',
        dest='bn_layer',
        help='Set whether batch_norm'
    )
    parser.set_defaults(bn_layer=True)

    activations = {'sigmoid': sigmoid,
                   'tanh': tanh,
                   'softplus': softplus,
                   'relu': rectify,
                   'elu': elu}

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

    train(options.data,
          activations[options.activation],
          options.plot_image,
          options.rseed,
          options.n_epoch,
          optimizers[options.optimizer],
          options.l,
          options.k,
          options.sample_l,
          options.sample_k,
          options.gamma,
          options.gan_gamma,
          options.sampling_type,
          options.n_batch,
          options.annealing,
          options.annealing_epoch,
          options.bn_layer,
          options_dict)
