from flax import linen as nn
import jax.numpy as jnp
from jax.tree_util import tree_map
import jax
from typing import Any, Callable, Sequence, Tuple
from functools import partial

ModuleDef = Any

activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}
pools = {'max_pool': nn.max_pool, 'avg_pool': nn.avg_pool}
    
def count_parameters(params):
    "counts the number of parameters of a model"
    return sum(x.size for x in jax.tree_leaves(params))

class muLinear(nn.Module):
    """ muP Readout layer """
    fan_out: int # num_classes
    use_bias: bool = True # bool for bias
    varw: float = 2.0 # variance
    
    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        # 1 / fan_out init
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'truncated_normal', mode = 'fan_out') 
        x = nn.Dense(self.fan_out, kernel_init = kernel_init, use_bias = self.use_bias)(x)
        # sqrt(fan_out / fan_in) multiplier
        x *= jnp.sqrt(self.fan_out / fan_in)
        return x

class muReadout(nn.Module):
    """ muP Readout layer """
    fan_out: int # num_classes
    use_bias: bool = True
    
    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        # 1 / fan_in initialization
        kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'truncated_normal', mode = 'fan_in') 
        x = nn.Dense(self.fan_out, kernel_init = kernel_init, use_bias = self.use_bias)(x)
        #  sqrt(1 / fan_in) multiplier
        x *= jnp.sqrt(1 / fan_in)
        return x

class Conv(nn.Module):
    """ Conv layer with fan_in initialziation """
    fan_out: int # number of filters
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1) # strides
    varw: float = 2.0 # variance
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        # varw / fan_in initialization
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'truncated_normal', mode = 'fan_in') 
        x = nn.Conv(self.fan_out, kernel_size = self.kernel_size, strides = self.strides, padding = 'SAME', kernel_init = kernel_init, use_bias = self.use_bias)(x)
        return x

class muConv(nn.Module):
    """ muP Conv layer with fan_out init and sqrt(fan_out / fan_in) """
    fan_out: int # number of filters
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1) # strides
    varw: float = 2.0 # variance
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[1] * x.shape[2] * x.shape[3]
        # varw / fan_out initialization
        kernel_init = nn.initializers.variance_scaling(scale = self.varw, distribution = 'truncated_normal', mode = 'fan_out') # kernel initialization
        x = nn.Conv(self.fan_out, kernel_size = self.kernel_size, strides = self.strides, padding = 'SAME', kernel_init = kernel_init, use_bias = self.use_bias)(x)
        # sqrt(fan_out / fan_in) multiplier
        x *= jnp.sqrt(self.fan_out / fan_in)
        return x

############################
########## UV model ########
############################

class UV_int(nn.Module):
    """
    Description: UV model that interpolates between NTP and muP 
    Inputs:
        1. width: width of the network
        2. out_dim: output dimension / number of classes
        3. varw: effective variance of each weights
        4. scale: [0, 1] interpolates between NTP and muP
    """
    width: int = 512
    out_dim: int = 1
    varw: float = 1.0
    scale: float = 1.0
    
    def setup(self):
        # setup initialization for all but last layer
        kernel_init = jax.nn.initializers.normal(stddev = jnp.sqrt(self.varw / self.width**self.scale) ) 
        # First layer
        self.U = nn.Dense(self.width, use_bias = False, kernel_init = kernel_init) 
        # Last layer
        self.V = nn.Dense(self.out_dim, use_bias = False, kernel_init = kernel_init) 
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1)) # flatten the input
        h = self.U(x)
        f = self.V(h) / jnp.sqrt(self.width**(1-self.scale))
        return jnp.asarray(f)

############################
########## FCNs ############
############################


class fcn_sp(nn.Module):
    """ FCNs in SP in He initialization"""
    width: int = 512
    depth: int = 4
    out_dim: int = 10
    use_bias: bool = False
    varw: float = 2.0
    act_name: str = 'relu'

    def setup(self):
        # setup initialization for all but last layer
        kernel_init = partial(jax.nn.initializers.variance_scaling, distribution = 'truncated_normal', mode = 'fan_in')
        # setup activation
        self.act = activations[self.act_name]
        # create a list of all but last layer
        self.layers = [nn.Dense(self.width, use_bias = self.use_bias, kernel_init = kernel_init(scale = self.varw)) for d in range(self.depth-1)]
        # last layer with different initialization constant
        lst_layer = [nn.Dense(self.out_dim, use_bias = self.use_bias, kernel_init = kernel_init(scale = 1.0)) ]
        # combine all layers
        self.layers += tuple(lst_layer)
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1)) # flatten the input to generalize to arbitrary shapes like images
        for d, layer in enumerate(self.layers):
            x = layer(x)
            if d+1 != self.depth:
                x = self.act(x)
        return jnp.asarray(x)
    

class fcn_mup(nn.Module):
    """ constant width muP FCNs """
    width: int = 512
    depth: int = 4
    out_dim: int = 10
    use_bias: bool = False
    varw: float = 2.0
    act_name: str = 'relu'

    def setup(self):
        # setup activation
        self.act = activations[self.act_name]
        # intermediate layers
        self.layers = tuple([muLinear(self.width, use_bias = self.use_bias, varw = self.varw) for d in range(self.depth-1)])
        # last layer with different initialization constant
        self.layers += tuple([muReadout(self.out_dim, use_bias = self.use_bias) ])
        return

    def __call__(self, inputs):
        x = inputs.reshape((inputs.shape[0], -1)) # reshape to generalize to all layers
        for d, layer in enumerate(self.layers):
            x = layer(x) # linear layer
            if d+1 != self.depth:
                x = self.act(x) # activation 
        return jnp.asarray(x)

###########################################
############## Myrtle CNNs ################
###########################################

class ConvBlock(nn.Module):
  """ Convolution block: Convolution followed by activation """
  Conv: ModuleDef
  filters: int
  act: Callable = nn.relu
  kernel_size: Tuple[int, int] = (3, 3)
  strides: Tuple[int, int] = (1, 1) # strides
  varw: float = 2.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x,):
    x = self.Conv(fan_out = self.filters, kernel_size = self.kernel_size, strides = self.strides, varw = self.varw, use_bias = self.use_bias)(x)
    x = self.act(x)
    return x

class Myrtle_sp(nn.Module):
    """ Myrtle CNNs in Standard Parameterization """
    num_filters: int # number of filters
    num_layers: int # number of layers
    num_classes: int # number of classes
    pool_list: Sequence[int] # pooling list
    Conv: ModuleDef = Conv
    kernel: Tuple[int, int] = (3, 3) # kernel size
    use_bias: bool = False # wheather to use bias or not
    act: Callable = nn.relu # activation
    varw: float = 2.0 # variance of the weights

    @nn.compact
    def __call__(self, x):        
        ## forward pass
        for i in range(self.num_layers):
            x = ConvBlock(self.num_filters, Conv = self.Conv, act = self.act, kernel = self.kernel)(x)
            if i in self.pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        # use mean to apply average pooling
        x = jnp.mean(x, axis = (1, 2)) 
        # last layer
        kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'truncated_normal', mode = 'fan_in') 
        x = nn.Dense(self.num_classes, use_bias = self.use_bias, kernel_init = kernel_init)(x)
        return jnp.asarray(x)
    
Myrtle5_sp = partial(Myrtle_sp, pool_list = [1, 2, 3], num_layers = 4)
Myrtle7_sp = partial(Myrtle_sp, pool_list = [1, 3, 5], num_layers = 6)
Myrtle10_sp = partial(Myrtle_sp, pool_list = [2, 5, 8], num_layers = 9)

class Myrtle_mup(nn.Module):
    """ Myrtle CNNs in Standard Parameterization """
    num_filters: int # number of filters
    num_layers: int # number of layers
    num_classes: int # number of classes
    pool_list: Sequence[int] # pooling list
    Conv: ModuleDef = Conv
    kernel: Tuple[int, int] = (3, 3) # kernel size
    use_bias: bool = False # wheather to use bias or not
    act: Callable = nn.relu # activation
    varw: float = 2.0 # variance of the weights

    @nn.compact
    def __call__(self, x):        
        ## forward pass
        for i in range(self.num_layers):
            x = ConvBlock(self.num_filters, Conv = self.Conv, act = self.act, kernel = self.kernel)(x)
            if i in self.pool_list:
                x = nn.avg_pool(x, (2 ,2) , (2 ,2))
        # use mean to apply average pooling
        x = jnp.mean(x, axis = (1, 2)) 
        # last layer
        x = muReadout(self.num_classes, use_bias = self.use_bias)(x)
        return jnp.asarray(x)

Myrtle5_mup = partial(Myrtle_mup, pool_list = [1, 2, 3], num_layers = 4)
Myrtle7_mup = partial(Myrtle_mup, pool_list = [1, 3, 5], num_layers = 6)
Myrtle10_mup = partial(Myrtle_mup, pool_list = [2, 5, 8], num_layers = 9)

######################################
############  WideResNets ############
######################################

class WideResNetBlock(nn.Module):
    """ WideResNet Block """
    features: int # number of filters
    Conv: ModuleDef
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1) # strides
    use_bias: bool = True
    act: Callable = nn.relu # activation
    varw: float = 2.0 # float
    
    @nn.compact
    def __call__(self, x):
        # First block
        y = self.Conv(self.features, kernel_size = self.kernel_size, strides = self.strides, varw = self.varw, use_bias = self.use_bias)(x)
        y = nn.LayerNorm()(y)
        y = self.act(y)
        # Seconf block
        y = self.Conv(self.features, kernel_size = self.kernel_size, varw = self.varw, use_bias = self.use_bias)(y)
        y = nn.LayerNorm()(y)
        # reshape the inputs x to have the same dimension as the output y
        if x.shape[-1] != self.features or self.strides != (1, 1):
            x = self.Conv(self.features, kernel_size = (1, 1), strides = self.strides, varw = 1.0, use_bias = self.use_bias)(x)
        # residual connection and then apply activation
        return self.act(y + x)

class WideResNet(nn.Module):
    """ WideResNet in SP """
    stage_sizes: Sequence[int]
    num_filters: int
    widening_factor: int
    num_classes: int
    Conv: ModuleDef = Conv
    kernel_size: Tuple[int, int] = (3, 3)
    use_bias: bool = True
    act: Callable = nn.relu # activation
    varw: float = 2.0
    
    @nn.compact
    def __call__(self, x):
        ## FORWARD PASS
        # First convolution + layernorm + activation
        x = self.Conv(self.num_filters, kernel_size = self.kernel_size, varw = self.varw, use_bias = self.use_bias)(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)

        # stages and blocks
        for stage, num_blocks in enumerate(self.stage_sizes):
            for block in range(num_blocks):
                features = self.num_filters * (2 ** stage) * self.widening_factor
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = WideResNetBlock(features, Conv = self.Conv, kernel_size = self.kernel_size, strides = strides, use_bias = self.use_bias, act = self.act, varw = self.varw)(x)
                
        # take a global average along axes [1, 2]
        x = jnp.mean(x, axis = (1, 2))
        kernel_init = nn.initializers.variance_scaling(scale = 1.0, distribution = 'truncated_normal', mode = 'fan_in') 
        x = nn.Dense(self.num_classes, kernel_init = kernel_init, use_bias = self.use_bias)(x) 
        return jnp.asarray(x)

WideResNet16_sp = partial(WideResNet, stage_sizes = [2, 2, 2])
WideResNet20_sp = partial(WideResNet, stage_sizes = [3, 3, 3])
WideResNet28_sp = partial(WideResNet, stage_sizes = [4, 4, 4])
WideResNet40_sp = partial(WideResNet, stage_sizes = [6, 6, 6])

class WideResNet_mup(nn.Module):
    """ WideResNet interpoaltes between SP and muP through the parameter scale """
    stage_sizes: Sequence[int]
    num_filters: int
    widening_factor: int
    num_classes: int
    Conv: ModuleDef = muConv
    kernel_size: Tuple[int, int] = (3, 3)
    use_bias: bool = True
    act: Callable = nn.relu # activation
    varw: float = 2.0
    
    @nn.compact
    def __call__(self, x):
        ## FORWARD PASS
        # First convolution + layernorm + activation
        x = self.Conv(self.num_filters, kernel_size = self.kernel_size, varw = self.varw, use_bias = self.use_bias)(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)

        # stages and blocks
        for stage, num_blocks in enumerate(self.stage_sizes):
            for block in range(num_blocks):
                features = self.num_filters * (2 ** stage) * self.widening_factor
                strides = (2, 2) if stage > 0 and block == 0 else (1, 1)
                x = WideResNetBlock(features, Conv = self.Conv, kernel_size = self.kernel_size, strides = strides, use_bias = self.use_bias, act = self.act, varw = self.varw)(x)
                
        # take a global average along axes [1, 2]
        x = jnp.mean(x, axis = (1, 2))
        x = muReadout(self.num_classes, use_bias = self.use_bias)(x) 
        return jnp.asarray(x)

WideResNet16_mup = partial(WideResNet_mup, stage_sizes = [2, 2, 2])
WideResNet20_mup = partial(WideResNet_mup, stage_sizes = [3, 3, 3])
WideResNet28_mup = partial(WideResNet_mup, stage_sizes = [4, 4, 4])
WideResNet40_mup = partial(WideResNet_mup, stage_sizes = [6, 6, 6])
