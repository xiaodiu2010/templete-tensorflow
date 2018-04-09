import tensorflow as tf
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from nets import sample_net
from nets import nets_factory
#from tensorlayer.cost import dice_coe
slim = tf.contrib.slim


class Unet(object):
    def __init__(self, config):
        self.config = config
        self.network = nets_factory.get_network_fn(self.config)
        if self.config.train.use_batch:
            self.normalizer_fn = slim.batch_norm
            self.batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': self.config.input.is_train
            }
            print("Using Batch Normalization")
        else:
            self.normalizer_fn = None
            self.batch_norm_params = None
            print("Not Using Batch Normalization")

    def net(self, images, reuse=False):
        '''
        with slim.arg_scope(net.arg_scope(weight_decay=self.config.train.weight_decay,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.batch_norm_params)):
        '''
        logits, end_points = self.network(images, self.config)
        print(end_points)
        with tf.variable_scope('feature_fusion', values=[end_points.values], reuse=reuse):
            with slim.arg_scope(self.arg_scopes()):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                for i in range(len(f)):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None for i in range(len(f))]
                h = g
                num_outputs = [None, 512, 256, 128, 128]
                for i in range(len(f)):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 3)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)

                        if self.config.network.use_deconv:
                            g[i] = slim.conv2d_transpose(h[i], num_outputs[i+1], 3, stride=[2,2])
                        else:
                            g[i] = slim.conv2d(self.unpool(h[i]), num_outputs[i+1], 2)
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

                g[-1] = slim.conv2d(g[-1], 64, 3)
                g[-1] = slim.conv2d(g[-1], 64, 3)
                f_score = slim.conv2d(g[-1], self.config.network.num_classes, 3,
                                      activation_fn=None,
                                      normalizer_fn=None)
                #f_score = tf.nn.softmax(f_score, -1)
                print(f_score)
        return f_score, end_points

    def arg_scopes(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(self.config.train.weight_decay),
                            #weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                data_format=self.config.input.data_format) as sc:
                return sc

    def loss(self, y_pred_cls, y_true_cls, type="dice"):
        # weighted loss
        # the count of each classes: [1.0729496e+09 5.6478020e+06 1.9082424e+07 2.4619714e+07 2.7321870e+07 4.8611740e+06]
        # coef [0.01, 1, ,1, ,1, ,1, 1]
        coef = np.ones(self.config.network.num_classes)
        coef[0] = 0.01
        if type=="dice":
            #y_pred_cls = tf.nn.softmax(y_pred_cls)
            eps = 1.
            intersection = tf.reduce_sum(y_true_cls * y_pred_cls, (0, 1, 2)) * coef
            union = tf.reduce_sum(y_true_cls * y_true_cls, (0, 1, 2)) * coef + \
                    tf.reduce_sum(y_pred_cls * y_pred_cls, (0, 1, 2)) * coef
            coef = (2 * intersection + eps)/ (union+eps)
            print(coef)
            loss = 1. - tf.reduce_mean(coef)
        else:
            y_true_cls = tf.argmax(y_true_cls, -1)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
            loss = tf.reduce_mean(loss)
        with tf.name_scope('total'):
            tf.add_to_collection('EXTRA_LOSSES', loss)
        tf.losses.add_loss(loss)
        return loss


    def unpool(self, inputs):
        _, h, w, c = inputs.get_shape().as_list()
        out = tf.image.resize_nearest_neighbor(inputs,
                                        size=[h * 2, w * 2],
                                        align_corners=True)
        out.set_shape([None, 2*h, 2*w, c])
        return out
