import tensorflow as tf
import os

def create_modest_session():
  # only using device 0
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"]="0"

  tf.reset_default_graph()
  # allowing GPU memory growth to allocate only what we need
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config, graph = tf.get_default_graph())
  return sess

def fc_layer(x, n, activation = tf.nn.relu):
    """ Fully connected layer for input x and output dim n """
    return tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True)

def concat_list(lst):
    """ Concatenate a list of tensors into a single 1D tensor """
    return tf.concat([tf.reshape(x, (-1, )) if x is not None else [0] for x in lst], 0)

def cos_similarity_vec(v1, v2, eps = 1e-7):
    """ Cosine similarity for vectors """
    return tf.tensordot(v1, v2, [[0], [0]]) / (tf.linalg.norm(v1) * tf.linalg.norm(v2) + eps)

def cos_similarity(at1, at2):
    """ Cosine similarity between two arrays of tensors of same length and shapes
        cosTheta = (a, b) / |a|_2|b|_2
    """
    # flattening lists of tensors
    at1_f = concat_list(at1)
    at2_f = concat_list(at2)
    return cos_similarity_vec(at1_f, at2_f)

def norm_fro_sq(x):
    """ Flatten and take squared norm """
    return tf.reduce_sum(tf.square(x))

def trainable_of(loss):
    """ Get trainable variables on which loss depends """
    return [x for x in tf.trainable_variables() if tf.gradients(loss, [x])[0] is not None]
