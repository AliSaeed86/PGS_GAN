# ---------------------------------------------------------
# Tensorflow Vessel-GAN (V-GAN) Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
'''
import os
import tensorflow as tf
from solver import Solver 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   
if physical_devices:
     tf.config.experimental.set_virtual_device_configuration(
         physical_devices[0],
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_interval', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg', 10, 'ratio of gan loss to seg loss, default: 10')
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('discriminator', 'pixel', 'type of discriminator [pixel|patch1|patch2|image], default: patch1')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'RimOneV3_singleGAN', 'dataset name [DRIVE|STARE|Drishti-GS1|RimOneV3], default: Drishti-GS1')    ## here we have to specify the folder name that containg the images of the dataset we need to use i.e., Drishti-GS1/DRIVE/STARE/
tf.flags.DEFINE_bool('is_test', False, 'default: False (train)')               # False for train and True for Test ## Added by Ali
tf.flags.DEFINE_bool('Use_both_GAN', False, 'default: True (to train both GANs)')             

tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
tf.flags.DEFINE_integer('iters', 11600, 'number of iteratons, default: 50000')    #11600 used in our training
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'models are saved here')
tf.flags.DEFINE_string('sample_dir', './sample', 'sample are saved here')
tf.flags.DEFINE_string('test_dir', './test', 'test images are saved here')
    
tf.flags.DEFINE_integer('train_interval_OC', 1, 'training interval between discriminator and generator, default: 1')
tf.flags.DEFINE_integer('ratio_gan2seg_OC', 10, 'ratio of gan loss to seg loss, default: 10')
tf.flags.DEFINE_string('discriminator_OC', 'pixel', 'type of discriminator [pixel|patch1|patch2|image], default: patch1')
tf.flags.DEFINE_string('checkpoint_dir_OC', './checkpoints', 'models are saved here')
tf.flags.DEFINE_string('sample_dir_OC', './sample', 'sample are saved here')
tf.flags.DEFINE_string('test_dir_OC', './test', 'test images are saved here')

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_test:
        solver.test()
    if not FLAGS.is_test:
        solver.train()

if __name__ == '__main__':
    tf.app.run()
