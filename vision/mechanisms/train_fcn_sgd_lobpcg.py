    # in use imports

import utils.model_utils as model_utils
import utils.train_utils as train_utils
import utils.data_utils as data_utils
import utils.loss_utils as loss_utils
import utils.schedules_utils as schedules_utils

import jax
from jax import numpy as jnp
import optax
from flax import linen as nn

from typing import Tuple

#usual imports
import pandas as pd
import argparse

# for deterministic gpu computations
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

"""Model definition and train state definition"""

def create_train_state(config: argparse.ArgumentParser, batch: Tuple):
    x, y = batch

    x = x[:config.batch_size, ...]
    y = y[:config.batch_size, ...]

    # create model
    model = models[config.model](width = config.width, depth = config.depth, out_dim = config.out_dim, use_bias = config.use_bias, varw = config.varw, act_name = config.act_name)

    # initialize using the init seed
    key = jax.random.PRNGKey(config.init_seed)
    init_params = model.init(key, x)['params']
    
    # debugging: check shapes and norms
    shapes = jax.tree_util.tree_map(lambda x: x.shape, init_params)
    print(shapes)
    norms = jax.tree_util.tree_map(lambda x: config.width * jnp.var(x), init_params)
    print(norms)

    # count the number of parameters
    num_params = model_utils.count_parameters(init_params)
    print(f'The model has {num_params/1e6:0.4f}M parameters')

    # create an optimizer
    opt = optax.inject_hyperparams(optax.sgd)(learning_rate = config.lr_init, momentum = config.momentum)
    state = train_utils.TrainState.create(apply_fn = model.apply, params = init_params, opt = opt)
    
    return state, num_params


def train_and_evaluate(config: argparse.ArgumentParser, train_ds: Tuple, test_ds: Tuple):
    "train model acording the config"
    
    
    # create train and test batches for measurements: measure batches are called train_batches and val_batches; training batches are called batches
    seed = config.sgd_seed
    rng = jax.random.PRNGKey(seed)

    train_loader = train_utils.data_stream(seed, train_ds, config.batch_size, augment = config.use_augment)
    test_loader = train_utils.data_stream(seed, test_ds, config.batch_size, augment = False)

    # create a train state
    state, num_params = create_train_state(config, train_ds)

    # prepare an initial guess for the eigenvectors of the hessian
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)
    key = jax.random.PRNGKey(93)
    vs_init = jax.random.normal(key, shape = (flat_params.shape[0], config.topk)) 
    # compute sharpness and update the target learning rate
    sharpness_init = train_utils.compute_sharpness_dataset(state, train_loader, config.loss_fn, vs_init, config.measure_batches)
    config.lr_trgt = config.c_trgt / sharpness_init
    print(f'Initial sharpness: {sharpness_init:0.4f}, Target learning rate: {config.lr_trgt:0.4f}')
    
    ########### TRAINING ##############
        
    # store training results
    train_results = list()
    eval_results = list()

    divergence = False

    running_loss = 0.0
    running_accuracy = 0.0
    
    lr_step = config.lr_init
    config.lr_min = config.lr_trgt / 10.0


    for step in range(config.num_steps):  

        epoch = (step // config.num_batches) + 1 
        cosine_step = state.step - config.warmup_steps + 1

        batch = next(train_loader)
        imgs, targets = batch

        # update the learning rate in the warmup phase
        if step < config.warmup_steps:
            lr_step = schedules_utils.polynomial_warmup(state.step+1, config.lr_init, config.lr_trgt, config.warmup_steps, exponent = config.warmup_exponent) # state.step + 1 used because there is not training step yet
        else:
            lr_step = schedules_utils.cosine_decay_schedule(cosine_step+1, config.lr_trgt, config.lr_min, config.num_steps - config.warmup_steps + 1, exponent = config.decay_exponent)

        state.update_learning_rate(learning_rate = lr_step)
        state, logits_step, grads_step, loss_step, sharpness_step, vs_step, n_iter = train_utils.train_sharpness_lobpcg_step(state, batch, config.loss_fn, vs_init)
        sharpness_step = sharpness_step.squeeze()

        accuracy_step = train_utils.compute_accuracy(logits_step, targets)

        result = jnp.array([state.step, epoch, lr_step, loss_step, accuracy_step, sharpness_step, n_iter])
        train_results.append(result)
        
        #check for divergence
        if (jnp.isnan(loss_step) or jnp.isinf(loss_step)): divergence = True; break

        running_loss += loss_step
        running_accuracy += accuracy_step
        
        #print(f'step: {state.step}, cosine step: {cosine_step}, lr_step: {lr_step:0.4f}, train loss: {loss_step:0.4f},  sharpness: {sharpness_step:0.4f}')

        if state.step % config.num_batches == 0:
            # compute sharpness 
            sharpness = train_utils.compute_sharpness_dataset(state, train_loader, config.loss_fn, vs_init, config.measure_batches)

            # estiamte the running loss and running accuracy; reset the parameters
            train_loss = running_loss / config.num_batches
            running_loss = 0.0

            train_accuracy = running_accuracy / config.num_batches
            running_accuracy = 0.0

            # estimate test accuracy
            test_loss, test_accuracy = train_utils.compute_eval_metrics_dataset(state, test_loader, config.loss_fn, config.num_test, config.batch_size)
            print(f't: {state.step}, lr_step: {lr_step:0.4f}, training loss: {train_loss:0.4f}, train_accuracy: {train_accuracy:0.4f}, test_loss: {test_loss:0.4f}, test_accuracy: {test_accuracy:0.4f}')
            result = jnp.asarray([state.step, epoch, lr_step, train_loss, train_accuracy, test_loss, test_accuracy, sharpness])
            eval_results.append(result)

    train_results = jnp.asarray(train_results)
    train_results = jax.device_get(train_results)

    eval_results = jnp.asarray(eval_results)
    eval_results = jax.device_get(eval_results)

    return divergence, train_results, eval_results, num_params

models = {'fcn_mup': model_utils.fcn_mup, 'fcn_sp': model_utils.fcn_sp}
loss_fns = {'mse': loss_utils.mse_loss, 'xent': loss_utils.cross_entropy_loss}
activations = {'relu': nn.relu, 'tanh': jnp.tanh, 'linear': lambda x: x}

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'nexus')
# Dataset parameters
parser.add_argument('--dataset', type = str, default = 'cifar-10')
parser.add_argument('--out_dim', type = int, default = 10)
parser.add_argument('--num_examples', type = int, default = 50000)

# Model parameters
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--width', type = int, default = 512)
parser.add_argument('--widening_factor', type = int, default = 1)
parser.add_argument('--depth', type = int, default = 4)
parser.add_argument('--bias', type = str, default = 'True') # careful about the usage
parser.add_argument('--act_name', type = str, default = 'relu')
parser.add_argument('--init_seed', type = int, default = 1)
parser.add_argument('--scale', type = float, default = 0.0)
parser.add_argument('--varw', type = float, default = 2.0)
#Optimization parameters
parser.add_argument('--loss_name', type = str, default = 'xent')
parser.add_argument('--augment', type = str, default = 'False')
parser.add_argument('--opt_name', type = str, default = 'sgd')
parser.add_argument('--sgd_seed', type = int, default = 1)
parser.add_argument('--warmup_steps', type = int, default = 512)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--decay_exponent', type = float, default = 0.0)
parser.add_argument('--num_steps', type = int, default = 2048)
parser.add_argument('--lr_exp', type = float, default = 0.0)
parser.add_argument('--lr_step', type = float, default = 1.0)
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--batch_size', type = int, default = 512)
# Sharpness estimation
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--sharpness_method', type = str, default = 'lobpcg')
parser.add_argument('--measure_batches', type = int, default = 10)
parser.add_argument('--tol', type = float, default = 1e-09)
parser.add_argument('--m_iter', type = int, default = 40)


config = parser.parse_args()

# Model parameters
config.model = f'fcn_{config.abc}'
config.use_bias = True if config.bias == 'True' else False
config.use_augment = True if config.augment == 'True' else False

# define loss
config.loss_fn = loss_fns[config.loss_name]

save_dir = 'fcn_results'

# Dataset loading 

if config.cluster == 'nexus':
    config.ds_dir = '/nfshomes/dayal/datasets'
elif config.cluster == 'zaratan':
    config.ds_dir = '/home/dayal/scratch.cmtc/datasets'
else:
    config.ds_dir = 'datasets'

(x_train, y_train), (x_test, y_test) = data_utils.load_image_data_tfds(config.dataset, flatten = False, subset = True, num_examples = config.num_examples)

config.num_train, config.num_test = x_train.shape[0], x_test.shape[0]

# standardize the inputs
x_train = data_utils._standardize(x_train, abc = 'sp')
x_test = data_utils._standardize(x_test, abc = 'sp')

config.in_dim = int(jnp.prod(jnp.array(x_train.shape[1:])))
config.out_dim = len(jnp.unique(y_train))

# one hot encoding for the labels
y_train = jax.nn.one_hot(y_train, config.out_dim)
y_test = jax.nn.one_hot(y_test, config.out_dim)

config.num_batches = max(100, train_utils.estimate_num_batches(config.num_train, config.batch_size))

print(config)

### TRAIN THE NETWORK AND EVALUATE ####
divergence = False
lr_exp = config.lr_exp

config.c_trgt = 2**lr_exp
divergence, train_results, eval_results, num_params = train_and_evaluate(config, (x_train, y_train), (x_test, y_test))
    
# create a dataframe
df_train = pd.DataFrame(train_results, columns = ['step', 'epoch', 'lr', 'loss_step', 'accuracy_step', 'sharpness_step', 'n_iter'])
df_train['num_params'] = num_params
# save training data
path = f'{save_dir}/train_{config.dataset}_{config.model}_scale{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lrexp{lr_exp:0.4f}_a{config.warmup_exponent}_b{config.decay_exponent}_Twarm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}_{config.sharpness_method}.tab'    
df_train.to_csv(path, sep = '\t')
            
if not divergence: 
    # create a dataframe
    # state.step, epoch, lr_step, train_loss_mean, train_loss_var, train_accuracy_mean, train_accuracy_var, test_loss_mean, test_loss_var, test_accuracy_mean, test_accuracy_var, sharpness_mean
    df_eval = pd.DataFrame(eval_results, columns = ['step', 'epoch', 'lr', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 'sharpness'])
    df_eval['num_params'] = num_params
    # save evaluation data
    path = f'{save_dir}/eval_{config.dataset}_{config.model}_scale{config.scale}_varw{config.varw}_n{config.width}_d{config.depth}_bias{config.use_bias}_{config.act_name}_I{config.init_seed}_{config.loss_name}_augment{config.augment}_{config.opt_name}_lrexp{lr_exp:0.4f}_a{config.warmup_exponent}_b{config.decay_exponent}_Twarm{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_m{config.momentum}_{config.sharpness_method}.tab'    
    df_eval.to_csv(path, sep = '\t')
