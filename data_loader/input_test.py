import os, sys
sys.path.append('./')
sys.path.append('../')
import tensorflow as tf
import datetime
from utils.config import process_config
from data_loader.data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np

config = '../configs/example.json'

config = process_config(config)
print(config)


def unpool(inputs, align_corners=True):
    _, h, w, c = inputs.get_shape().as_list()
    out = tf.image.resize_bilinear(inputs,
                                           size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2],
                                           align_corners=align_corners)
    # out = out.set_shape([None, h, w, c])
    return out

dataset_train = DataGenerator(config.input)

x_train, y_train = dataset_train.get_train_data()
x_train.set_shape([None, config.input.out_shape[0], config.input.out_shape[1], 3])
y_train.set_shape([None, config.input.out_shape[0], config.input.out_shape[1]])
print(x_train)
x_resize_align = unpool(x_train)
x_resize_no_ali = unpool(x_train, align_corners=False)

y_train_hot = tf.one_hot(y_train, depth=config.network.num_classes)

scaffold = tf.train.Scaffold(
    init_op=None,
    init_feed_dict=None,
    init_fn=None,
    ready_op=None,
    ready_for_local_init_op=None,
    local_init_op=dataset_train.get_iterator().initializer,
    summary_op=None,
    saver=None,
    copy_from_scaffold=None
)

sess = tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=scaffold,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=600,
        save_summaries_steps=100,
        save_summaries_secs=None,
        config=None,
        stop_grace_period_secs=120,
        log_step_count_steps=100
)

step = 0
visual = True
y_stat = []
while not sess.should_stop():
    start_time = datetime.datetime.now()
    x, y, x_align, x_no_ali, y_hot = sess.run([x_train, y_train, x_resize_align, x_resize_no_ali, y_train_hot])
    #print("Loaded {} examples using {} seconds".format(x.shape[0],datetime.datetime.now()-start_time))
    step += 1
    #y_hot_stat = np.sum(y_hot, (0,1,2))
    #print(y_hot_stat.shape)
    #y_stat.append(np.expand_dims(y_hot_stat,0))

    if visual:
        # Visualization
        if step % 10 == 0:
            f, axs = plt.subplots(4, 4, figsize=(16, 16))
            for i in range(x.shape[0]):
                img, mask, img_align, img_no_ali = x[i], y[i], x_align[i], x_no_ali[i]
                plt.subplot(4, 4, 4*i+1)
                plt.imshow(img.astype(np.uint8))
                plt.subplot(4, 4, 4*i+2)
                plt.imshow(mask.astype(np.uint8))
                plt.subplot(4, 4, 4*i+3)
                plt.imshow(img_align.astype(np.uint8))
                plt.subplot(4, 4, 4*i+4)
                plt.imshow(img_no_ali.astype(np.uint8))
            plt.show()



