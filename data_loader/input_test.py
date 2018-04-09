import os, sys
sys.path.append('./')
sys.path.append('../')
import tensorflow as tf
import datetime
from utils.config import process_config
from data_loader.data_generator import DataGenerator
import matplotlib.pyplot as plt
import numpy as np

config = '../configs/test.json'

config = process_config(config)
print(config)


def unpool(inputs, align_corners=True):
    _, h, w, c = inputs.get_shape().as_list()
    out = tf.image.resize_bilinear(inputs,
                                           size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2],
                                           align_corners=align_corners)
    # out = out.set_shape([None, h, w, c])
    return out

dataset = DataGenerator(config.input)

x_train, y_train = dataset.get_train_data()
x_train.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
y_train.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1]])


x_test, y_test = dataset.get_eval_data()
x_test.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
y_test.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1]])

print(x_train)
#print(y_test)

scaffold = tf.train.Scaffold(
    init_op=None,
    init_feed_dict=None,
    init_fn=None,
    ready_op=None,
    ready_for_local_init_op=None,
    local_init_op=[dataset.get_iterator().initializer,
                   dataset.get_iterator(is_train=False).initializer],
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
    x, y, x_t, y_t = sess.run([x_train, y_train, x_test, y_test])
    #x, y = sess.run([x_train, y_train])
    #print("Loaded {} examples using {} seconds".format(x.shape[0],datetime.datetime.now()-start_time))
    step += 1

    print("max:{} min:{}".format(x.max(), x.min()))
    print(x.shape)
    num = 1
    if visual:
        # Visualization
        if step % 10 == 0:
            f, axs = plt.subplots(num, 4, figsize=(num*4, 16))
            for i in range(num):
                img, mask, x_timg, y_timg = x[i], y[i], x_t[i], y_t[i]
                plt.subplot(num, 4, 4*i+1)
                plt.imshow(img.astype(np.uint8))
                plt.subplot(num, 4, 4*i+2)
                plt.imshow(mask.astype(np.uint8))
                plt.subplot(num, 4, 4*i+3)
                plt.imshow(x_timg.astype(np.uint8))
                plt.subplot(num, 4, 4*i+4)
                plt.imshow(y_timg.astype(np.uint8))
            plt.show()



