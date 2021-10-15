from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow.compat.v1  as tf #HW: Updated to include compat
tf.disable_v2_behavior()
file = 'C://Users//hlw69//Documents//fMRI_transferlearning//Zhang et al//ICW-GANs-master//ICW-GANs-master//checkpoint//1952_ori.model-20000'
filename='C://Users//hlw69//Documents//fMRI_transferlearning//Zhang et al//ICW-GANs-master//ICW-GANs-master//checkpoint//1952_ori.model-20000.meta'
print_tensors_in_checkpoint_file(file_name=file, tensor_name='', all_tensors=False)



with tf.Session() as sess:
    saver = tf.train.import_meta_graph(filename)
    saver.restore(sess,file)
   # print(sess.run('w1:0'))