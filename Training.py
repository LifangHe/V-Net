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
tf.app.flags.DEFINE_integer('iterations',10,
    """Number of iterations for training""")
tf.app.flags.DEFINE_integer('test_each',1000,
    """Test each n-th iteration.""")
tf.app.flags.DEFINE_string('log_dir', './tmp/log',
    """Directory where to write training and testing event logs """)
tf.app.flags.DEFINE_float('init_learning_rate',1e-2,
    """Initial learning rate""")
tf.app.flags.DEFINE_float('decay_factor',0.99,
    """Exponential decay learning rate factor""")
tf.app.flags.DEFINE_integer('decay_steps',500,
    """Number of iterations before applying one learning rate decay""")
tf.app.flags.DEFINE_integer('display_step',1000,
    """Display and logging interval (train steps)""")
tf.app.flags.DEFINE_integer('save_interval',1000,
    """Checkpoint save interval (iterations)""")
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


def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3, 4], smooth=1e-5):

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
    dice = (tf.constant(2.0) * tf.cast(inse,dtype=tf.float32) + tf.constant(smooth)) / (tf.cast(l + r, dtype=tf.float32) + tf.constant(smooth))
    dice = tf.reduce_mean(dice)
    return dice


"""Train the V-Net model"""

with tf.Graph().as_default():
    
    global_step = tf.train.get_or_create_global_step()

    # Data
    with tf.name_scope("data"):
        # dictionary for sampling
        params = {f.params[i]: f.params[i + 1] for i in range(0, len(f.params), 2)}

        training_data = ThreadedDataSetCollection(f.w, f.p, f.data_location, f.train_folder, f.files, f.masks, f.nclasses, params)
        test_data = ThreadedDataSetCollection(f.w, f.p, f.data_location, f.test_folder, f.files, f.masks, f.nclasses)

        input_batch_shape = training_data.get_shape()
        output_batch_shape = training_data.get_target_shape()
        
        images_placeholder, labels_placeholder = placeholder_inputs(input_batch_shape, output_batch_shape)


    # Model
    with tf.name_scope("vnet"):
        model = VNet(
                num_classes=f.nclasses, # 2 for binary
                keep_prob=f.drop_out, # default 1.0
                num_channels=16, # default 16 
                num_levels=4,  # default 4
                num_convolutions=(1,2,3,3), # default (1,2,3,3), size should equal to num_levels
                bottom_convolutions=3, # default 3
                activation_fn="prelu") # default relu

        logits = model.network_fn(images_placeholder) # (n_batches, nx, nz, nz, n_classes)


    # Softmax to generate label from logits
    with tf.name_scope("predicted_label"):
        pred = tf.nn.softmax(logits) # (n_batches, nx, nz, nz, n_classes)


    # Learning rate
    with tf.name_scope("learning_rate"):
        learning_rate = tf.train.exponential_decay(f.init_learning_rate, global_step, 
            f.decay_steps, f.decay_factor, staircase=False)
    tf.summary.scalar('learning_rate', learning_rate)


    # Loss
    with tf.name_scope("loss"):
        loss_function = f.loss_function
        if (loss_function == "xent"):
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels_placeholder)) # (n_batches, nx, ny, nz) -> ()
        elif (loss_function == "sorensen"):
            loss_op = 1.0-dice_coe(pred,tf.cast(labels_placeholder,dtype=tf.float32), loss_type='sorensen')
        elif (loss_function == "jaccard"):
            loss_op = 1.0-dice_coe(pred,tf.cast(labels_placeholder,dtype=tf.float32), loss_type='jaccard')
    tf.summary.scalar('loss',loss_op)


    # Metrics
    with tf.name_scope("metrics"):
        tp, tp_op = tf.metrics.true_positives(labels_placeholder, tf.round(pred), name="true_positives")
        tn, tn_op = tf.metrics.true_negatives(labels_placeholder, tf.round(pred), name="true_negatives")
        fp, fp_op = tf.metrics.false_positives(labels_placeholder, tf.round(pred), name="false_positives")
        fn, fn_op = tf.metrics.false_negatives(labels_placeholder, tf.round(pred), name="false_negatives")
        sensitivity_op = tf.divide(tf.cast(tp_op,tf.float32),tf.cast(tf.add(tp_op,fn_op),tf.float32))
        specificity_op = tf.divide(tf.cast(tn_op,tf.float32),tf.cast(tf.add(tn_op,fp_op),tf.float32))
        dice_op = 2.*tp_op/(2.*tp_op+fp_op+fn_op)
        correct_pred = tf.equal(tf.cast(tf.round(pred),dtype=tf.int64), tf.cast(labels_placeholder,dtype=tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('sensitivity', sensitivity_op)
    tf.summary.scalar('specificity', specificity_op)
    tf.summary.scalar('dice', dice_op)


    # Training Op
    with tf.name_scope("training"):
        # optimizer
        optimizer = f.optimizer
        momentum = f.momentum
        if optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == "nesterov_momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)

        train_op = optimizer.minimize(loss=loss_op, global_step=global_step)

        # Update op is required by batch norm layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])


    # # iteration checkpoint manipulation
    start_iteration = tf.get_variable("start_iteration", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
    start_iteration_inc = start_iteration.assign(start_iteration+1)


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
        sess.run(tf.local_variables_initializer())
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
        
            print("{}: Last checkpoint iteration: {}".format(datetime.datetime.now(),start_iteration.eval()[0]))
            print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(sess, global_step)))


        # Loop over iterations
        for iteration in np.arange(start_iteration.eval(), f.iterations):
            print("{}: iteration {} starts.".format(datetime.datetime.now(),iteration+1))

            # training phase
            model.is_training = True
            image, label = training_data.random_sample()
            train, summary, loss = sess.run([train_op, summary_op, loss_op], feed_dict={images_placeholder: image, labels_placeholder: label})
            train_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))

            if ((iteration+1) % f.test_each) == 0:

                batches = test_data.get_volume_batch_generators()
                model.is_training = False
                for batch, file, shape, w, p in batches:
                    for image, label, imin, imax in batch:
                        summary, loss = sess.run([summary_op, loss_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                        test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))


        start_iteration_inc.op.run()
        # print(start_iteration.eval())
        # save the model at end of each iteration training
        print("{}: Saving checkpoint of iteration {} at {}...".format(datetime.datetime.now(),iteration+1,f.checkpoint_dir))
        if not (os.path.exists(f.checkpoint_dir)):
            os.makedirs(f.checkpoint_dir,exist_ok=True)
        saver.save(sess, checkpoint_prefix, 
            global_step=tf.train.global_step(sess, global_step),
            latest_filename="checkpoint-latest")
        print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))
            
        # testing phase
        print("{}: Training of iteration {} finishes, testing start".format(datetime.datetime.now(),iteration+1))
                
        model.is_training = False

        batches = test_data.get_volume_batch_generators()

        for batch, file, shape, w, p in batches:
            for image, label, imin, imax in batch:
                summary, loss = sess.run([summary_op, loss_op], feed_dict={images_placeholder: image, labels_placeholder: label})
                test_summary_writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step))


    # close tensorboard summary writer
    train_summary_writer.close()
    test_summary_writer.close()