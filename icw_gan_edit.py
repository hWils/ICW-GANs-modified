# coding=utf-8

import os, sys

# tensorflow 1
sys.path.append(os.getcwd())

import time
import argparse
import matplotlib
import tkinter

matplotlib.use('TkAgg')  # HW: changed from 'agg'
import numpy as np
import tensorflow.compat.v1  as tf  # HW: Updated to include compat

tf.disable_v2_behavior()  # HW: Updated to ensur default to variable v1, which includes collections rather than v2

import tflib as lib
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.BrainPedia
import tflib.plot
import tflib.ops.conv3d
from tflib.upsampling import *
from six.moves import xrange
import nibabel
import pickle as pkl
from nilearn.input_data import NiftiMasker

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint",
                    help="Directory name to save the checkpoints")
parser.add_argument('--tags_ignore', type=str, default="36, 37, 34, 44, 38, 28, 23, 33, 32",
                    help="ignore_tags")

parser.add_argument("--cond", type=str, default="110101", help="conditional")
parser.add_argument("--concat", type=str, default="1", help="concate method")
parser.add_argument("--or_shape", type=str, default="[53,63,46]",
                    help="original image shape")  # HW downscaled is [53,63,46] upscaled is [79, 95, 69],
parser.add_argument("--gpu", type=str, default="0", help="gpu_id")

parser.add_argument("--multi", type=int, default="6", help="times of penalty")
parser.add_argument("--sample_dir", type=str, default="./samples/", help="sample")
parser.add_argument("--test_dir", type=str, default="./synthetic/", help="test")
parser.add_argument("--cost_dir", type=str, default="./costs/", help="cost")

parser.add_argument("--base_dir", type=str, default="./pkl/", help="base_dir to put training pkl data")
parser.add_argument("--imageFile", type=str, default="original_dim.pkl",
                    help="imageFile")  # not necerssary for new load method
parser.add_argument("--labelFile", type=str, default="multi_class_pic_tags.pkl",
                    help="labelFile")  # not necerssary for new load method

parser.add_argument("--MODE", type=str, default="wgan-gp", help="the model being used")
parser.add_argument("--DIM", type=int, default="64", help="model dimensionality")
parser.add_argument("--BATCH_SIZE", type=int, default="10", help="BATCH_SIZE")  # HW: was originally 50
parser.add_argument("--CRITIC_ITERS", type=int, default="5",
                    help="For WGAN and WGAN-GP, number of critic iters per gen iter")
parser.add_argument("--LAMBDA", type=int, default="10", help="Gradient penalty lambda hyperparameter")
parser.add_argument("--ITERS", type=int, default="1010",
                    help="How many generator iterations to train for")  # HW: was originally 200,000: 5-10 epochs.
parser.add_argument("--y_dim", type=int, default="2", help="The number of classes")  # HW: was originally 45

# args = parser.parse_args()
FLAGS = parser.parse_args()
print(FLAGS)

tags_leave_out = []
for tag in FLAGS.tags_ignore.strip().split(','):
    tags_leave_out.append(int(tag))

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

if not os.path.exists(FLAGS.test_dir):
    os.makedirs(FLAGS.test_dir)
if not os.path.exists(FLAGS.cost_dir):
    os.makedirs(FLAGS.cost_dir)

MODE = FLAGS.MODE
DIM = FLAGS.DIM
BATCH_SIZE = FLAGS.BATCH_SIZE
CRITIC_ITERS = FLAGS.CRITIC_ITERS
LAMBDA = FLAGS.LAMBDA
ITERS = FLAGS.ITERS
y_dim = FLAGS.y_dim

shape = FLAGS.or_shape[1:-1].split(',')
shape = [int(i) for i in shape]

OUTPUT_DIM = 1  # Number of pixels in MNIST (28*28)
for s in shape:
    OUTPUT_DIM = OUTPUT_DIM * s
OUTPUT_DIM = int(OUTPUT_DIM)

lib.print_model_settings(locals().copy())

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    y_shapes = y.get_shape()
    x_shapes = x.get_shape().as_list()
    x = concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]])], 4)
    return x


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


# produces samples, takes as input random noise
def Generator(n_samples, y=None, noise=None, origin_shape=None, layers=5, Reuse=False, is_training=True):
    shape_ori = origin_shape[1:-1].split(',')  # 53,69,46
    shape_ori = [int(i) for i in shape_ori]
    shape = [shape_ori[0], shape_ori[1], shape_ori[2]]
    # print("Our shape is ", shape)

    sp_layers = []
    sp_layers.append(shape)
    for i in range(layers):
        sp_layers.append([int(i / 2 + i % 2) for i in sp_layers[-1]])  # 46     # our dimensions [79,95,69]
    # 1,
    if noise is None:
        noise = tf.random.normal([n_samples, 128])  # HW: changed from tf.random_normal
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])
    for i in range(layers):
        if (i == 0):
            if (FLAGS.cond[0] == '1'):
                noise = concat([noise, y], 1)
            noise_shape = noise.get_shape().as_list()
            rshp = sp_layers[-1][0] * sp_layers[-1][1] * sp_layers[-1][2] * min(2 ** (layers - 1), 8) * DIM
            output = lib.ops.linear.Linear('Generator.Input', noise_shape[-1], rshp, noise)
            output = lib.ops.batchnorm.BN('Generator.BN' + str(i), -1,
                                          output, Reuse=Reuse, is_training=is_training)
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, sp_layers[-1][2], sp_layers[-1][0],
                                         sp_layers[-1][1], min(2 ** (layers - 1), 8) * DIM])
            # print('G0: ', output.get_shape().as_list())
        else:

            if (FLAGS.cond[i] == '1'):
                if (FLAGS.concat == '0'):  # HW: we want this to be set to zero,
                    output = conv_cond_concat(output, yb)
                else:
                    sp = output.get_shape().as_list()
                    yb1_ = lib.ops.linear.Linear('Generator.yb' + str(i) + '_', y_dim, sp[1] * sp[2] * sp[3] * 3, y)
                    yb1_ = tf.tanh(yb1_)
                    yb1_ = tf.reshape(yb1_, [-1, sp[1], sp[2], sp[3], 3])
                    output = concat([output, yb1_], 4)
            output = lib.ops.conv3d.Deconv('Generator.' + str(i), output.get_shape().as_list()[-1],
                                           [BATCH_SIZE, sp_layers[-(i + 1)][2], sp_layers[-(i + 1)][0],
                                            sp_layers[-(i + 1)][1],
                                            min(2 ** (layers - i - 1), 8) * DIM], output)

            output = lib.ops.batchnorm.BN('Generator.BN' + str(i), -1, output, Reuse=Reuse, is_training=is_training)
            output = tf.nn.relu(output)
            # print('G: ', i, output.get_shape().as_list())
    # [ batchsize, x,y,z , ?] conv2d  Labels
    output = lib.ops.conv3d.Deconv('Generator.' + str(layers), output.get_shape().as_list()[-1],
                                   [BATCH_SIZE, sp_layers[0][2], sp_layers[0][0], sp_layers[0][1], 1], output)
    output = tf.nn.sigmoid(output)
    # print('last shape in G: ',output.shape)
    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output


def Discriminator(inputs, y=None, origin_shape=None, layers=5, Reuse=None):
    # default :"NDHWC"
    shape = origin_shape[1:-1].split(',')
    shape = [int(i) for i in shape]
    output = tf.reshape(inputs, [-1, shape[2], shape[0], shape[1], 1])
    # output = inputs
    sp_layers = []
    sp_layers.append(shape)
    for i in range(layers):
        sp_layers.append([i / 2 + i % 2 for i in sp_layers[-1]])

    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 1, y_dim])

    for i in range(layers, 0, -1):  # 4~0

        if (FLAGS.cond[i] == '1'):
            if (FLAGS.concat == '0'):
                output = conv_cond_concat(output, yb)
            else:
                sp = output.get_shape().as_list()
                yb4_ = lib.ops.linear.Linear('Discriminator.yb' + str(i) + '_', y_dim, sp[1] * sp[2] * sp[3] * 3, y)
                yb4_ = tf.reshape(yb4_, [-1, sp[1], sp[2], sp[3], 3])
                yb4_ = tf.tanh(yb4_)
                output = concat([output, yb4_], 4)
        output = lib.ops.conv3d.Conv3D('Discriminator.' + str(i), output.get_shape().as_list()[-1],
                                       min(2 ** (layers - i), 8) * DIM, output, stride=2)
        output = LeakyReLU(output)
        # print('D ', i, output.get_shape().as_list())

    sp = output.get_shape().as_list()
    output = tf.reshape(output, [sp[0], -1])

    if (FLAGS.cond[0] == '1'):
        output = concat([output, y], 1)

    sp = output.get_shape().as_list()
    output = lib.ops.linear.Linear('Discriminator.0', sp[1], 512, output)
    output = lib.ops.linear.Linear('Discriminator.Output', 512, 1, output)
    output = tf.reshape(output, [-1])
    print(" Output form discriminator is : ", output)
    # print('last output of D ', output.get_shape().as_list())
    # weighted  output = output* prop
    return output


tf.compat.v1.disable_eager_execution()  # HW: added
real_data = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
if y_dim:
    y = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, y_dim], name='y')
else:
    y = None

# prop = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE], name='prop')

fake_data = Generator(BATCH_SIZE, y, noise=None, origin_shape=FLAGS.or_shape)  # HW: shape is imposed here
disc_real = Discriminator(real_data, y, origin_shape=FLAGS.or_shape, Reuse=None)
disc_fake = Discriminator(fake_data, y, origin_shape=FLAGS.or_shape, Reuse=True)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(input_tensor=disc_fake)
    disc_cost = tf.reduce_mean(input_tensor=disc_fake) - tf.reduce_mean(input_tensor=disc_real)

    gen_train_op = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.compat.v1.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':  # hw: wasserstein gans with gradient penalty (WGAN-GP)

    # evenly distributed
    gen_cost = -tf.reduce_mean(input_tensor=disc_fake)  # -closs
    # unevenly distributed
    disc_cost = tf.reduce_mean(input_tensor=disc_fake) - tf.reduce_mean(
        input_tensor=disc_real)  # - closs #（tf.multiply(disc_real, prop))

    # dis_cost_1 = tf.reduce_mean(disc_real)
    # dis_cost_2 = tf.reduce_mean(tf.multiply(disc_real, prop))
    alpha = tf.random.uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)

    gradients = tf.gradients(ys=Discriminator(interpolates,
                                              y, origin_shape=FLAGS.or_shape, Reuse=True), xs=[interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients), axis=[1]))
    gradient_penalty = tf.reduce_mean(input_tensor=(slopes - 1.) ** 2)
    disc_cost += LAMBDA * gradient_penalty

    global_step = tf.Variable(  # HW: changed Variable => VariableV1
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.compat.v1.GraphKeys.GLOBAL_STEP, tf.compat.v1.GraphKeys.GLOBAL_VARIABLES])

    learning_rate = tf.compat.v1.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step,
        decay_steps=10000 * 2,
        decay_rate=0.5,
        staircase=True)

    # 1e-4,
    learning_rate2 = tf.compat.v1.train.exponential_decay(
        learning_rate=1e-4,
        global_step=global_step,
        decay_steps=10000 * 2,
        decay_rate=0.5,
        staircase=True)
    '''
    disc_train_op = tf.train.GradientDescentOptimizer(
        learning_rate=1e-4
    ).minimize(disc_cost, var_list=disc_params)

    '''
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gen_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_cost, var_list=gen_params, global_step=global_step)

        disc_train_op = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-4,
            beta1=0.5,
            beta2=0.9
        ).minimize(disc_cost, var_list=disc_params, global_step=global_step)

    clip_disc_weights = None
    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.compat.v1.train.Saver()
#	saver = tf.train.import_meta_graph('my_test_model-1000.meta')

elif MODE == 'dcgan':  # hw: deep convolutional gan
    gen_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.ones_like(disc_fake)
    ))

    disc_cost = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real,
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=1e-3,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=1e-3,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
# data generators, which shuffles, float32, int32
train_gen, dev_gen, test_gen = lib.BrainPedia.new_load_data(BATCH_SIZE,
                                                            FLAGS.base_dir,
                                                            FLAGS.imageFile,
                                                            FLAGS.labelFile,
                                                            tags_leave_out=tags_leave_out,
                                                            y_dim=y_dim)
choose_pool = []
for i in range(y_dim):
    if (i not in tags_leave_out):
        choose_pool.append(i)

ran = np.random.choice(choose_pool, BATCH_SIZE)
fixed_labels = np.zeros((BATCH_SIZE, y_dim), dtype=np.float64)  # used in generate image
for i, label in enumerate(ran):
    print("Ran is ", i, label, fixed_labels[i, ran[i]])

    fixed_labels[i, ran[i]] = 1.0
print("fixed labels are ", fixed_labels)
fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
fixed_noise_samples = Generator(BATCH_SIZE, y, noise=fixed_noise, origin_shape=FLAGS.or_shape, Reuse=True,
                                is_training=False)


def generate_image(frame):
    samples = session.run(fixed_noise_samples, feed_dict={y: fixed_labels})
    samples_b = samples.reshape([BATCH_SIZE, shape[2], shape[0], shape[1]])
    samples_b = np.transpose(samples_b, [0, 2, 3, 1])

    count = 0
    for i in range(BATCH_SIZE):

        img = nibabel.Nifti1Image(samples_b[i, :, :, :],
                                  np.eye(4))  # HW: removed mask as not using one here msk.affine)
        #	plt.show()
        # zero-out
        #	temp = msker.transform(img)
        #	braindata = msker.inverse_transform(temp)  # WHAT DOES THIS DO HW
        id = list(fixed_labels[count]).index(1.)  # this is the 2nd or [0,1]
        filename = './{}/samples{}_{}_{}.nii.gz'.format(FLAGS.sample_dir, frame, count, id)
        nibabel.save(img, filename)
        count += 1
        if count == 5:
            break


# TODO: find out from Peiye what this does
def have_mask_affine():
    msk_file = open(FLAGS.base_dir + 'msk_p2.pkl', 'rb')
    msk = pkl.load(msk_file)
    msk_file.close()
    msker = NiftiMasker(mask_img=msk, standardize=False)
    return msk, msker


# HW: commented out mask affine
# msk, msker = have_mask_affine()
# msker.fit()

# all generated images, i don't think there are any real ones
def save_test_img(frame):
    choose_pool = []
    for i in range(y_dim):
        if (i not in tags_leave_out):
            choose_pool.append(i)
    #	print("These are the tags ", choose_pool) #HW
    ran = np.random.choice(choose_pool, BATCH_SIZE)
    #	print("the randomly selected tags are ", ran) #HW
    test_labels = np.zeros((BATCH_SIZE, y_dim), dtype=np.float64)
    for i, label in enumerate(ran):
        test_labels[i, ran[i]] = 1.0
    #	print("The test labels assigned are ", test_labels)

    test_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    test_noise_samples = Generator(BATCH_SIZE, y, noise=test_noise, origin_shape=FLAGS.or_shape, Reuse=True,
                                   is_training=False)
    print("Running test session now")  # HW
    samples = session.run(test_noise_samples, feed_dict={y: test_labels})  # 10 , 2
    #	print("  y in save_test_img is ", y)

    samples = samples.reshape([BATCH_SIZE, shape[2], shape[0], shape[1]])
    samples = np.transpose(samples, [0, 2, 3, 1])  # HW: the parameter samples was originally samples_b

    count = 0
    for i in range(BATCH_SIZE):
        img = nibabel.Nifti1Image(samples[i, :, :, :], np.eye(4))  # msk.affine)
        # zero-out
        # temp = msker.transform(img)
        # braindata = msker.inverse_transform(temp)
        #	print("id  ", list(test_labels[i]), list(test_labels[i]).index(1.))
        id = list(test_labels[i]).index(1.)
        if id == 0:
            condition = 'place'
        elif id == 1:
            condition = 'face'
        print("frame is ", frame, ". count is ", count, ". id is ", id, " . This means condition is ", condition)
        filename = './{}/test_synth_{}_{}_{}{}.nii.gz'.format(FLAGS.test_dir, frame, count, id)
        nibabel.save(img, filename)
        count += 1


def inf_train_gen():
    while True:
        for images, targets in train_gen():
            yield images, targets  # returns a generator


def save(checkpoint_dir, sess, step=0):
    model_name = "brian.model"
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load(sess, checkpoint_dir):
    all_vars = tf.global_variables()
    print("Actual variables are :")
    for var in all_vars:
        print(var.name)

    print("Should now be looking for a checkpoint")
    import re
    import slim
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # train tensor
    print("ckpt ", ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print('[INFO] CKPT: ', ckpt, ckpt.model_checkpoint_path)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        # saver = tf.train.Saver(tf.compat.v1.slim.get_variables_to_restore(exclude ='Discriminator.0/Discriminator.0.W' ))
        # HW:  attempt to exclude the non existent variables to restore
        model_vars = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        checkpoint_vars = tf.train.list_variables(checkpoint_dir)
        #	print("The intersection ", set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(
        #	tf.train.list_variables(checkpoint_dir)))
        exclude_vars = 'Discriminator.0/Discriminator.0'
        exclude_filter = 'Discriminator.5/Discriminator.5.Filters'
        exclude_bias = 'Discriminator.yb'
        exclude_gen_bias = 'Generator.yb'
        exclude = 'Generator.1/Generator.1'
        exl = 'Generator.BN0'
        vars = [item for item in model_vars if exclude_vars not in item.name]
        vars = [item for item in vars if exclude_filter not in item.name]
        vars = [item for item in vars if exclude_bias not in item.name]
        vars = [item for item in vars if exclude_gen_bias not in item.name]
        vars = [item for item in vars if exclude not in item.name]
        vars = [item for item in vars if exl not in item.name]
        for name in vars:
            print(name.name)
        print("Vars are ", vars)
        tf.variables_initializer('Discriminator.0/Discriminator.0.W')
        global_vars = tf.global_variables('Discriminator.0/Discriminator.0.W')
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        print("Not initialised ", is_not_initialized)
        saver = tf.train.Saver(vars)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in vars])
        print("Not initialised ", is_not_initialized)
        # init_ckpt = tf.contrib.framework.assign_from_checkpoint_fn(checkpoint_dir, vars)
        # init_ckpt(sess)
        # variables_can_be_restored = list(set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(
        #	tf.train.list_variables(checkpoint_dir)))
        #	temp_saver = tf.train.Saver(variables_can_be_restored)
        #	ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir, lastest_filename=None)
        # print('Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
        #	temp_saver.restore(sess, ckpt_state.model_checkpoint_path)

        # saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))  # FAILURE POINT??

        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def load_ckpt(sess, ckpt):
    print(" [*] Reading checkpoints...")
    saver.restore(sess, ckpt)


MAX_FRACTION = 0.7
NUM_THREADS = 2
sess_config = tf.compat.v1.ConfigProto()
sess_config.gpu_options.allow_growth = True  #
sess_config.intra_op_parallelism_threads = NUM_THREADS
sess_config.gpu_options.per_process_gpu_memory_fraction = MAX_FRACTION  #

# Train loop
with tf.compat.v1.Session(config=sess_config) as session:
    _, ct = load(session, FLAGS.checkpoint_dir)
    try:
        tf.global_variables_initializer().run()  # compat.v1.
    except:
        print("Does it get into the exception")
        tf.compat.v1.initialize_all_variables().run()
    print("Finished with checkpoints")
    gen = inf_train_gen()
    writer = tf.summary.FileWriter('./graphs', graph=session.graph)  # HW: Added
    for iteration in xrange(ITERS):  # How many generator iterations to train for
        print("iteration ", iteration)
        start_time = time.time()
        # get generator loss
        if iteration > 0:
            for i in xrange(3):  # xrange uses yielding, but similar to range
                _, g_loss = session.run([gen_train_op, gen_cost], feed_dict={y: targets})  # feed_dic are the inputs
            print('iter:%d,  d_loss: %.4f,  g_loss: %.4f' % (iteration, _disc_cost, g_loss))
            lib.plot.plot(FLAGS.cost_dir + '/Training cost of G', g_loss)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        _data, targets = next(gen)  # get next set of images and labels
        temp_prop = []
        # get discriminator cost
        _disc_cost, _ = session.run(
            [disc_cost, disc_train_op],
            feed_dict={real_data: _data, y: targets})

        if clip_disc_weights is not None:
            _ = session.run(clip_disc_weights)

        if (iteration > 0):
            lib.plot.plot(FLAGS.cost_dir + '/Training cost of D', _disc_cost)
            lib.plot.plot(FLAGS.cost_dir + '/time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 0 and iteration != 0:

            dev_disc_costs = []
            dev_gen_costs = []

            # validation every 100 iterations
            for images, tags in dev_gen():  # image and targets
                temp_prop = []

                _dev_disc_cost, _dev_g_loss = session.run([disc_cost, gen_cost],
                                                          feed_dict={real_data: images, y: tags}
                                                          )
                dev_disc_costs.append(_dev_disc_cost)
                dev_gen_costs.append(_dev_g_loss)
            lib.plot.plot(FLAGS.cost_dir + '/Developing cost of D', np.mean(dev_disc_costs))
            lib.plot.plot(FLAGS.cost_dir + '/Developing cost of G', np.mean(dev_gen_costs))

        # Write logs every 100 iters
        if (iteration < 5 and iteration > 0) or (iteration % 100 == 99):
            lib.plot.flush(FLAGS.cost_dir)
        lib.plot.tick()

        if np.mod(iteration, 5000) == 0:
            print('[INFO] Save checkpoint...')
        #	save(FLAGS.checkpoint_dir, session, iteration) #HWL commented out, do not want to overwrite at the moment

        if (iteration % 1000 == 0):  # HW: was originally 1000
            generate_image(iteration)
        # HW: they dont have a python testing file, they use this to save images
        if (iteration > 10000):  # originally 20000
            save_test_img(iteration)
        if (iteration > 11000):  # originally 20500
            print('over')
            break
writer.flush()  # HW: added
writer.close()  # HW: added
