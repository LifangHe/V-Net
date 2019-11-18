__author__ = "Georgi Tancev"
__copyright__ = "Copyright (C) 2019 Georgi Tancev"

from model.Model import VNet
from datasampling.DataSetCollection import ThreadedDataSetCollection
import numpy as np
import tensorflow as tf
import sys
import os
import math
import datetime

# select gpu devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # e.g. "0,1,2", "0,2" 


# tensorflow app flags
f = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_location', './files',
    """Directory of stored data.""")
tf.app.flags.DEFINE_string('train_folder', './train',
    """Directory of training data.""")
tf.app.flags.DEFINE_string('test_folder', './test',
    """Directory of training data.""")
tf.app.flags.DEFINE_list('files',['t2_pp.nii','pd_pp.nii','mprage_pp.nii','flair_pp.nii'],
    """Image filenames""")
tf.app.flags.DEFINE_list('masks',['mask.nii'],
    """Label filenames""")
tf.app.flags.DEFINE_integer('nclasses',2,
    """Label filenames""")
tf.app.flags.DEFINE_list('params',[],
    """Data augmentation parameters. First method, then parameter, e.g. --params ['param', 'value']""")
tf.app.flags.DEFINE_float('drop_out',0.5,
    """Probabiliy for DropOut""")               
tf.app.flags.DEFINE_multi_integer('w',[80,80,80],
    """Size of a subvolume""")
tf.app.flags.DEFINE_multi_integer('p',[5,5,5],
    """Padding of a subvolume""")
tf.app.flags.DEFINE_integer('epochs',10000,
    """Number of epochs for training""")
tf.app.flags.DEFINE_integer('test_each',1000,
    """Test each n-th epoch.""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.01,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',100,
    """Number of epoch before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',1000,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1000,
    """Checkpoint save interval (epochs)""")
tf.app.flags.DEFINE_string('checkpoint_dir', './tmp/ckpt',
    """Directory where to write checkpoint""")
tf.app.flags.DEFINE_string('model_dir','./tmp/model',
    """Directory to save model""")
tf.app.flags.DEFINE_bool('restore_training',False,
    """Restore training from last checkpoint""")
tf.app.flags.DEFINE_integer('shuffle_buffer_size',5,
    """Number of elements used in shuffle buffer""")
tf.app.flags.DEFINE_string('loss_function','jaccard',
    """Loss function used in optimization (xent, sorensen, jaccard)""")
tf.app.flags.DEFINE_string('optimizer','adam',
    """Optimization method (sgd, adam, momentum, nesterov_momentum)""")
tf.app.flags.DEFINE_float('momentum',0.5,
    """Momentum used in optimization""")

"""Define some helper functions"""

def placeholder_inputs(input_batch_shape, output_batch_shape):

    """Generate placeholder variables to represent the the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded ckpt in the .run() loop, below.
    Args:
        patch_shape: The patch_shape will be baked into both placeholders.
    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """

    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test ckpt sets.
    # batch_size = -1

    images_placeholder = tf.placeholder(tf.float32, shape=input_batch_shape, name="images_placeholder")
    labels_placeholder = tf.placeholder(tf.int32, shape=output_batch_shape, name="labels_placeholder")   

    return images_placeholder, labels_placeholder


def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    inse = tf.reduce_sum(tf.multiply(output,target), axis=axis)

    if loss_type == 'jaccard':
        l = tf.reduce_sum(tf.multiply(output,output), axis=axis)
        r = tf.reduce_sum(tf.multiply(target,target), axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
    ##
    dice = tf.reduce_mean(dice)
    return dice


"""Train the Vnet model"""

with tf.Graph().as_default():
    
    global_step = tf.train.get_or_create_global_step()

    # dictionary for sampling
    params = {f.params[i]: f.params[i + 1] for i in range(0, len(f.params), 2)}

    training_data = ThreadedDataSetCollection(f.w, f.p, f.data_location, f.train_folder, f.files, f.masks, f.nclasses, params)
    test_data = ThreadedDataSetCollection(f.w, f.p, f.data_location, f.test_folder, f.files, f.masks, f.nclasses)

    input_batch_shape = training_data.get_shape()
    output_batch_shape = training_data.get_target_shape()
    
    images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape, output_batch_shape)

    with tf.name_scope("vnet"):
        model = VNet(
                num_classes=f.nclasses, # binary for 2
                keep_prob=f.drop_out, # default 1
                num_channels=16, # default 16 
                num_levels=4,  # default 4
                num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
                bottom_convolutions=3, # default 3
                activation_fn="prelu") # default relu

        logits = model.network_fn(images_placeholder)

    with tf.name_scope("learning_rate"):
        learning_rate = f.init_learning_rate
    tf.summary.scalar('learning_rate', learning_rate)


    with tf.name_scope("softmax"):
        softmax_op = tf.nn.softmax(logits,name="softmax")


    with tf.name_scope("cross_entropy"):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=labels_placeholder))
    tf.summary.scalar('loss',loss_op)


    # Argmax Op to generate label from logits
    with tf.name_scope("predicted_label"):
        pred = tf.argmax(logits, axis=4 , name="prediction")


    # Accuracy of model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.expand_dims(pred,-1), tf.cast(labels_placeholder, dtype=tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    # Dice Similarity, currently only for binary segmentation
    with tf.name_scope("dice"):
        sorensen = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0], depth=2), dtype=tf.float32), loss_type='sorensen', axis=[1,2,3,4])
        jaccard = dice_coe(softmax_op,tf.cast(tf.one_hot(labels_placeholder[:,:,:,:,0], depth=2), dtype=tf.float32), loss_type='jaccard', axis=[1,2,3,4])
        sorensen_loss = 1. - sorensen
        jaccard_loss = 1. - jaccard
    tf.summary.scalar('sorensen', sorensen)
    tf.summary.scalar('jaccard', jaccard)
    tf.summary.scalar('sorensen_loss', sorensen_loss)
    tf.summary.scalar('jaccard_loss',jaccard_loss)


    # Training Op
    with tf.name_scope("training"):
        # optimizer
        optimizer = f.optimizer
        init_learning_rate = f.init_learning_rate
        momentum = f.momentum
        if optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=init_learning_rate)
        elif optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
        elif optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=init_learning_rate, momentum=momentum)
        elif optimizer == "nesterov_momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=init_learning_rate, momentum=momentum, use_nesterov=True)
        else:
            sys.exit("Invalid optimizer")

        # loss function
        loss_function = f.loss_function
        if (loss_function == "xent"):
            loss_fn = loss_op
        elif(loss_function == "sorensen"):
            loss_fn = sorensen_loss
        elif(loss_function == "jaccard"):
            loss_fn = jaccard_loss
        else:
            sys.exit("Invalid loss function")

        train_op = optimizer.minimize(
            loss = loss_fn,
            global_step=global_step)


    # # epoch checkpoint manipulation
    start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
    start_epoch_inc = start_epoch.assign(start_epoch+1)


    # saver
    summary_op = tf.summary.merge_all()
    checkpoint_prefix = os.path.join(f.checkpoint_dir ,"checkpoint")
    print("Setting up Saver...")
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4


    # training cycle
    with tf.Session(config=config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        print("{}: Start training...".format(datetime.datetime.now()))

        # summary writer for tensorboard
        train_summary_writer = tf.summary.FileWriter(f.log_dir + '/train', sess.graph)
        test_summary_writer = tf.summary.FileWriter(f.log_dir + '/test', sess.graph)

        # restore from checkpoint
        if f.restore_training:
            # check if checkpoint exists
            if os.path.exists(checkpoint_prefix+"-latest"):
                print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),f.checkpoint_dir))
                latest_checkpoint_path = tf.train.latest_checkpoint(f.checkpoint_dir,latest_filename="checkpoint-latest")
                saver.restore(sess, latest_checkpoint_path)
        
        print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval()[0]))
        print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))

        # loop over epochs
        for epoch in np.arange(start_epoch.eval(), f.epochs):
            print("{}: Epoch {} starts".format(datetime.datetime.now(),epoch+1))

            # training phase
            model.is_training = True
            image, label = training_data.random_sample()
            train, summary = sess.run([train_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
            train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

            if ((epoch+1) % f.test_each) == 0:

                batches = test_data.get_volume_batch_generators()

                for batch, file, shape, w, p in batches:
                    for image, label, imin, imax in batch:
                        loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))


        start_epoch_inc.op.run()
        # print(start_epoch.eval())
        # save the model at end of each epoch training
        print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,f.checkpoint_dir))
        if not (os.path.exists(f.checkpoint_dir)):
            os.makedirs(f.checkpoint_dir,exist_ok=True)
        saver.save(sess, checkpoint_prefix, 
            global_step=tf.train.global_step(sess, global_step),
            latest_filename="checkpoint-latest")
        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
            
        # testing phase
        print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
                
        model.is_training = False

        batches = test_data.get_volume_batch_generators()

        for batch, file, shape, w, p in batches:
            for image, label, imin, imax in batch:
                loss, summary = sess.run([loss_op, summary_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))


    # close tensorboard summary writer
    train_summary_writer.close()
    test_summary_writer.close()