import numpy as np

#xavier method used for weight initalization when tanh activation function is used
def xavier(input_nodes, output_nodes):
    return np.sqrt((1 / (output_nodes + input_nodes)))

#he method used for weight init when relu activation function is used
def he(input_nodes, output_nodes):
    return np.sqrt((2 / (output_nodes + input_nodes)))

def normal(input_nodes, output_nodes):
    return 1

def init_bias(weights):
    bias = []
    for layer in weights:
        bias.append(np.zeros([1, layer.shape[1]])) #Initialize bias to zero as default
    return bias

#Shape of each weight matrix consists of firstlayernodesXfollowinglayernodes
def init_weights(layers, input_nodes, output_nodes, method):
    weights = []
    first = None
    for layer in layers:
        if first != None:
            weights.append(np.random.normal(0, method(input_nodes, output_nodes), (first, layer.shape[0]))) #init with mean 0 and standard deviation 1 (normal distribution)
        first = layer.shape[0]
    return weights

#Here we follow pyramid structure with input layers as base, making each following layer smaller
def layer_length(input_nodes, output_nodes, deep_layers):
    _range = range(input_nodes, output_nodes -1, -1)
    base_percentile = 100 / deep_layers
    percentile = base_percentile
    yield input_nodes
    while percentile < 100:
        next_layer_nodes = np.percentile(_range, 100 - percentile).astype(np.int64)
        yield next_layer_nodes
        percentile += base_percentile
    yield output_nodes

def init_layers(deep_layers, input_nodes, output_nodes):
    layers = []
    for layer in layer_length(input_nodes, output_nodes, deep_layers):
        layers.append(np.zeros([layer, 1]),)
    return layers

def copy_object_shape(copy):
    ret = []
    for cpy in copy:
        ret.append(np.zeros(cpy.shape))
    return ret
