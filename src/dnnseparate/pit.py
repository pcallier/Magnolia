#!/usr/bin/env python
'''
Permutation-invariant cost for audio source separation.
Based on Kolbaek et al. (2017) manuscript. Replicates the CNN and dense
network from the paper, alongside an alternative, smaller CNN architecture.

Permutation-invariant training (PIT) aims to overcome the limitation that the outputs
of a multi-output neural network are order-sensitive, but the target truth is not. Any
permutation of outputs that corresponds to a valid separation of speakers should
be permitted by the objective function.

PIT calculates all pairwise assignments of output to unique target, and chooses
the minimum total loss over permutations of assignments during training.
'''
import sys
from itertools import islice, permutations, product

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from ..features.mixer import FeatureMixer
from ..features.wav_iterator import batcher
from ..features.spectral_features import scale_spectrogram
from ..utils.tf_utils import scope_decorator as scope

class PITModel:
    def __init__(self, method='pit-s-cnn', num_srcs=2,
        num_steps=50, num_freq_bins=513, learning_rate=0.001):
        '''
        Args:
            method (str): one of {'pit-s-cnn','pit-s-cnn-small', or 'pit-s-dnn'}.
                Selects the network architecture to use with the PIT loss
            num_srcs (int): number of sources to output reconstructions for. Inference
                on different numbers of speakers from the number the network was
                trained on is not yet implemented.
            num_steps (int): number of time steps to expect in the TF representation of
                input audio exemplars
            num_freq_bins (int): number of frequency bins in input spectrograms
            learning_rate (float): learning rate for optimizer (Adam)
        '''
        self.num_steps = num_steps
        self.num_freq_bins = num_freq_bins
        self.num_srcs = num_srcs
        self.X_in = tf.placeholder(tf.float32, (None, num_steps, num_freq_bins))
        self.y_in = tf.placeholder(tf.float32, (None, num_srcs, num_steps, num_freq_bins))
        self.learning_rate = learning_rate

        # Choose appropriate ops for the desired network architecture
        if method=='pit-s-cnn':
            self.network = self.cnn_mask
        elif method=='pit-s-cnn-small':
            self.network = self.cnn_mask_smaller
        elif method=='pit-s-dnn':
            self.network = self.dense_mask
        else:
            raise ValueError("Invalid network: {method}".format(method=method))

        # Register tensor operations as object attributes
        self.network
        self.mask
        self.logits
        self.predict
        self.loss
        self.optimize

    def load(self, path, sess=None):
        '''
        Load weights into the model graph from path.

        Args:
            path - path to Tensorflow checkpoint
            sess - Tensorflow session; if None, will be assigned to the default
                session
        '''
        if sess is None:
            sess = tf.get_default_session()
        # with sess.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, path)

    def save(self, path, sess=None):
        '''
        Save weights from the model graph at path.

        Args:
            path - path to Tensorflow checkpoint
            sess - Tensorflow session; if None, will be assigned to the default
                session
        '''
        if sess is None:
            sess = tf.get_default_session()
        # with sess.as_default():
        saver = tf.train.Saver()
        saver.save(sess, path)

    @scope
    def loss(self):
        '''
        Compute PIT loss across all examples in a minibatch,
        respecting that each example in the minibatch might
        have a different optimal mapping of truth to reconstruction

        Returns:
            TensorFlow op describe the minibatch loss of the model
        '''

        # compute pairwise costs
        # in 2-src case: either 1->2,2->1 or 1->1,2->2
        losses = []
        print("Losses")
        for src_id, out_id in product(range(self.num_srcs), range(self.num_srcs)):
            loss = tf.reduce_mean(self.X_in * tf.squared_difference(self.predict[:, out_id],
                                                        self.y_in[:, src_id]),
                                 axis=(1,2))
            losses.append(loss)

        # for each *permutation* of src->output assignments, look up the
        # appropriate losses and sum
        permuted_losses = []
        for assignment in permutations(range(self.num_srcs), self.num_srcs):
            permuted_loss = []
            for src_id, out_id in enumerate(assignment):
                loss_idx = src_id * self.num_srcs + out_id
                permuted_loss.append(losses[loss_idx])
            # sum losses over assigned pairings and add to list of permuted losses
            permuted_loss = tf.stack(permuted_loss, axis=1)
            permuted_loss = tf.reduce_sum(permuted_loss, axis=1)
            permuted_losses.append(permuted_loss)

        # select the minimum loss across assignment permutations
        permuted_losses = tf.stack(permuted_losses,axis=1)
        cost = tf.reduce_min(permuted_losses,axis=1)
        cost = tf.reduce_sum(cost)
        return cost

    @scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    @scope
    def predict(self):
        return self.network[0]

    @scope
    def mask(self):
        return self.network[1]

    @scope
    def logits(self):
        return self.network[2]

    @scope
    def dense_mask(self):
        '''
        Replicates PIT-S-DNN architecture from Kolbaek et al. 2017
        '''
        print("Dense (PIT-S-DNN) layers")
        x = flatten(self.X_in)
        x = tf.layers.dense(x, 1024, tf.nn.relu)
        x = tf.layers.dense(x, 1024, tf.nn.relu)
        x = tf.layers.dense(x, 1024, None)
        x = self.mask_ops(x)
        return x

    def mask_ops(self, x):
        '''
        Calculate and apply num_srcs masks from a dense transformation of the input
        tensors. Uses softmax to ensure masks sum to one across src id.
        '''
        # Predict mask
        print("Masks")
        all_masks = tf.layers.dense(x, self.num_srcs*self.num_steps*self.num_freq_bins, None,
            # kernel_initializer=tf.random_normal_initializer(stddev=0.001)
            )
        print(all_masks)
        all_masks = tf.reshape(all_masks, [-1, self.num_srcs, self.num_steps, self.num_freq_bins])
        print(all_masks)
        # Batch#,Mask#,T,F -> Mask#,Batch#,T,F for broadcasting across masks
        all_masks = tf.transpose(all_masks, [1,0,2,3])
        print(all_masks)
        all_masks = tf.exp(all_masks) / tf.reduce_sum(tf.exp(all_masks), 0)
        print(all_masks)

        # Reconstruct
        reconstructions = all_masks * self.X_in
        # Mask#,Batch#,T,F -> Batch#,Mask#,T,F
        reconstructions = tf.transpose(reconstructions, [1, 0, 2, 3])
        print("reconstructions")
        print(reconstructions)

        return reconstructions, all_masks, x

    @scope
    def cnn_mask_smaller(self):
        '''
        Shorter version of PIT-S-CNN (see cnn_mask)
        '''

        x = tf.expand_dims(self.X_in, 3)
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(2,2),
                             name='conv1', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(1,1),
                             name='conv2', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), strides=(2,2),
                             name='conv6', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), strides=(2,2),
                             name='conv9', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME', name='maxpool11')
        x = flatten(x)
        x = tf.layers.dense(x, 1024, activation=None, name='dense12')
        return self.mask_ops(x)

    @scope
    def cnn_mask(self):
        '''
        Attempts to replicate PIT-S-CNN architecture from Kolbaek et al. 2017
        Parameter count is high and I have had trouble training it so far.
        '''

        x = tf.expand_dims(self.X_in, 3)
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(2,2),
                             name='conv1', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(1,1),
                             name='conv2', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(1,1),
                             name='conv3', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(1,1),
                             name='conv4', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, kernel_size=(3, 3), strides=(1,1),
                             name='conv5', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), strides=(2,2),
                             name='conv6', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), strides=(1,1),
                             name='conv7', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 128, kernel_size=(3, 3), strides=(1,1),
                             name='conv8', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), strides=(2,2),
                             name='conv9', padding='SAME', activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), strides=(1,1),
                             name='conv10', padding='SAME', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, kernel_size=(3, 3), strides=(1,1),
                             name='conv11', padding='SAME', activation=tf.nn.relu)
        x = tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME', name='maxpool11')
        x = flatten(x)
        x = tf.layers.dense(x, 1024, activation=None, name='dense12')


        return self.mask_ops(x)

    @scope
    def blstm_mask(self):
        raise NotImplementedError("Haven't finished BLSTM yet")
        # x = flatten(self.X_in)
        #
        # fwd_lstm = tf.contrib.rnn.BasicLSTMCell(896)
        #
        # x = tf.nn.bidirectional_dynamic_rnn()
        # return self.mask_ops(x)

    def separate(self, mixture, sess=None):
        '''
        Perform separation on TF mixtures of arbitrary length, with
        overlap-and-add (triangular windows)

        Args:
            mixture: *one* input example (no support for batches right now),
                will be separated. Time x Frequency
            sess: tensorflow session
        '''

        window_length = self.num_steps
        step_length = self.num_steps // 2
        # where window_length is odd, operations involving
        # step_length on one side need step_length+1 on the other
        step_length_complement = step_length+window_length%2
        orig_mix_length = mixture.shape[0]
        spec_dtype = np.float32

        if sess is None:
            sess = tf.get_default_session()

        if orig_mix_length == self.num_steps:
            return sess.run(self.predict, { self.X_in: np.expand_dims(mixture, 0) })

        # Length of step must evenly divide into mixture length-window length
        length_to_pad = step_length - (orig_mix_length - window_length) % step_length
        zeros_to_pad = np.zeros((length_to_pad, self.num_freq_bins))
        mixture = np.concatenate((mixture, zeros_to_pad), axis=0)
        mix_length = orig_mix_length + length_to_pad

        # Window out input mixture, fill in reconstruction
        output_spectrograms = np.zeros((self.num_srcs, *mixture.shape), dtype=spec_dtype)
        window = np.bartlett(window_length)
        window_starts = range(0, mix_length-window_length+1, step_length)

        for win_start in window_starts:
            win_end = win_start + window_length
            mix_slice = np.array(mixture[win_start:win_end], dtype=np.float32)
            mix_slice = mix_slice.reshape(1, self.num_steps, self.num_freq_bins)
            # get spectrograms
            output_slice = sess.run(self.predict, {self.X_in: mix_slice})
            # window output for each source and add to output
            for src_id in range(self.num_srcs):
                output_spectrograms[src_id, win_start:win_end] += \
                    (output_slice[0, src_id].T * window).T

        # take care of head and tail of reconstruction (not overlapped properly)
        mix_head = np.concatenate((
            np.zeros((step_length_complement, self.num_freq_bins),
            dtype=mixture.dtype),
            mixture[:step_length]), axis=0)
        mix_tail = np.concatenate((
            mixture[:step_length],
            np.zeros((step_length_complement, self.num_freq_bins),
            dtype=mixture.dtype)), axis=0)
        output_head = sess.run(self.predict,
            {self.X_in: np.expand_dims(mix_head,0)})
        output_tail = sess.run(self.predict,
            {self.X_in: np.expand_dims(mix_tail,0)})
        for src_id in range(self.num_srcs):
            # add reconstruction of head to output
            output_spectrograms[src_id, :step_length] += \
                (output_head[0, src_id, step_length_complement:].T
                * window[step_length_complement:]).T

        return output_spectrograms[:, :orig_mix_length]


def training_setup(num_srcs, num_steps, num_freq_bins):
    '''
    Utility function for running training from the command line
    '''
    tf.reset_default_graph()

    model = PITModel('pit-s-dnn', num_srcs, num_steps, num_freq_bins)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    return model, features, references, sess

def training_loop(model, source_a, source_b, sess,
                  num_steps=51, num_freq_bins=257, batch_size=128, num_batches=10000):
    '''
    Implements a simple training loop for PIT and returns the losses.
    '''
    losses = []
    num_srcs = 2
    mixed_features = FeatureMixer([source_a, source_b], shape=(num_steps, None))
    data_batches = batcher(mixed_features, batch_size)

    for i, batch in enumerate(islice(data_batches, num_batches)):
        # Unpack batch
        batch_features, batch_ref1, batch_ref2 = batch
        # Normalize
        batch_ref1_norm, batch_ref1_norm_phase = scale_spectrogram(batch_ref1)
        batch_ref2_norm, batch_ref2_norm_phase = scale_spectrogram(batch_ref2)
        batch_features_norm, batch_features_norm_phase = scale_spectrogram(batch_features)
        data = {
            model.X_in: batch_features_norm,
            model.y_in: np.stack((batch_ref1_norm, batch_ref2_norm), axis=1)
            }
        sess.run(model.optimize, feed_dict=data)
        loss = sess.run(model.loss, feed_dict=data)
        losses.append(loss)
        prediction = sess.run(model.predict, data)

    return losses

# Running from command line mostly a toy application
if __name__=="__main__":
    try:
        src1, src2, num_batches = sys.argv[1:]
        num_batches = int(num_batches)
    except:
        print("Usage: pit src1 src2 num_batches\n"
            "\tsrc1, src2: paths to hdf5 files of TF signals to mix\n"
            "\tnum_batches: number of training iterations to run", file=sys.stderr)
        sys.exit(-1)
    model, features, references, sess = training_setup(2, 51, 257)
    losses = training_loop(model, src1, src2, features, references, sess, 51, 257, 64, num_batches)
    saver = tf.train.Saver()
    checkpoint_path = saver.save(sess, './pit_trained.ckpt')
    print("Model saved at {checkpoint_path}".format(checkpoint_path=checkpoint_path))
    with open("pit_losses.txt", "w") as f:
        for loss in losses:
            print(loss, file=f)
    print("Losses saved at 'pit_losses.txt'")
