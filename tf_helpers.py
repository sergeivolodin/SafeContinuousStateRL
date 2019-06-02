import tensorflow as tf
import os
import numpy as np

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

def fc_layer(x, n, activation = tf.nn.relu, name = 'fc'):
    """ Fully connected layer for input x and output dim n """
    return tf.identity(tf.contrib.layers.fully_connected(x, n, activation_fn=activation,
    weights_initializer=tf.initializers.lecun_normal(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, trainable=True), name = name)

def concat_list(lst):
    """ Concatenate a list of tensors into a single 1D tensor """
    return tf.concat([tf.reshape(x, (-1, )) if x is not None else [0] for x in lst], 0)

def cos_similarity_vec(v1, v2, eps = 1e-7):
    """ Cosine similarity for vectors """
    return tf.tensordot(v1, v2, [[0], [0]]) / (tf.linalg.norm(v1) * tf.linalg.norm(v2) + eps)

def cos_similarity(at1, at2, name = 'cos_sim'):
    """ Cosine similarity between two arrays of tensors of same length and shapes
        cosTheta = (a, b) / |a|_2|b|_2
    """
    # flattening lists of tensors
    at1_f = concat_list(at1)
    at2_f = concat_list(at2)
    return tf.identity(cos_similarity_vec(at1_f, at2_f), name = name)

def norm_fro_sq(x):
    """ Flatten and take squared norm """
    return tf.reduce_sum(tf.square(x))

def trainable_of(loss):
    """ Get trainable variables on which loss depends """
    return [x for x in tf.trainable_variables() if tf.gradients(loss, [x])[0] is not None]

def cond(pred, x, y):
    """ Conditional expression """
    return tf.cond(pred, lambda: x, lambda: y)

def dz_dw_flatten(z, params):
    """ Calculate dz/dparams and flatten the result """
    return tf.concat([tf.reshape(x, shape = (-1,)) for x in tf.gradients(z, params)], axis = 0)

def iterate_flatten(tensor):
    """ Iterate over flattened items of a tensor """
    if type(tensor) == list:
        for t in tensor:
            for v in iterate_flatten(t):
                yield v
    elif len(tensor.shape) == 0:
        yield tensor
    else:
        for idx in range(tensor.shape[0]):
            for v in iterate_flatten(tensor[idx]):
                yield v
                
def tf_hessian(var, params):
    # gradients of the loss w.r.t. params
    grads = tf.gradients(var, params)
    grad_components = list(iterate_flatten(grads))
    hessian = [dz_dw_flatten(t, params) for t in (grad_components)]
    return hessian

class OwnGradientDescent():
    def __init__(self, gamma = 0.5, theta = 0.9):
        # gamma (learning rate)
        self.gamma = tf.Variable(gamma, dtype = tf.float32)
        self.theta = theta
        
    def minimize(self, loss, params):
        """ Minimize some loss """
        def decrement_weights(W, gamma, grads):
            """ w = w - how_much """
            ops = [w.assign(tf.subtract(w, tf.multiply(gamma, grad))) for w, grad in zip(W, grads)]
            return tf.group(ops)
        
        # gradients of the loss w.r.t. params
        grads = tf.gradients(loss, params)
        
        # perform gradient descent step
        train_op = decrement_weights(params, self.gamma, grads)
        
        # updating gamma
        upd_op = self.gamma.assign(tf.multiply(self.gamma, self.theta))
        
        return tf.group(train_op, upd_op)

def CatVariable(shapes, initializer):
    """ List of tensors from a single tensor
    https://github.com/afqueiruga/afqstensorflowexamples/blob/master/afqstensorutils.py
    """

    l = np.sum([np.prod(shape) for shape in shapes])
    # V = tf.Variable(tf.zeros(shape=(l,)))

    V = tf.Variable(initializer(shape=(l,), dtype=tf.float64), dtype=tf.float64)

    cuts = []
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        cuts.append(tf.reshape(V[l:(l+il)],shp))
        l += il
    return V, cuts

class FCModelConcat():
    """ Fully-connected network with all weights in one tensor """
    def __init__(self, layer_shapes, activation = tf.nn.relu, initializer = tf.random.truncated_normal):
        """ Initialize with N_neurons (w/o input layer) """
        self.layer_shapes = layer_shapes

        # list of all shapes required
        self.shapes = []
        for i in range(len(layer_shapes) - 1):
            self.shapes.append((layer_shapes[i], layer_shapes[i + 1]))
            self.shapes.append((layer_shapes[i + 1],))
#        print(self.shapes)

        self.activation = activation

        # creating tensors...
        self.W, self.tensors = CatVariable(self.shapes, initializer = initializer)

        # filling weights and biases
        self.biases = []
        self.weights = []

        for i in range(len(self.tensors) // 2):
            self.weights.append(self.tensors[2 * i])
            self.biases.append(self.tensors[2 * i + 1])

    def forward(self, l0):
        # layers
        with tf.name_scope('layers'):
            # flattening the input
            z = tf.reshape(l0, (-1, np.prod(l0.shape[1:])))
            i_max = len(self.weights) - 1
            for i in range(i_max + 1):
                z = z @ self.weights[i] + self.biases[i]
                if i < i_max:
                    z = self.activation(z)
            return z
