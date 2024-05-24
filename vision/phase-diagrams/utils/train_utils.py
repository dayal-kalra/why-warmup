#Some imports
import jax
import jax.numpy as jnp
import optax
from typing import Any, Callable, Sequence, Tuple
from functools import partial
from flax import core
from flax import struct
from jax.numpy.linalg import norm
from jax.experimental import sparse
import numpy as np
import utils.data_utils as data_utils

class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node = False)
    params: core.FrozenDict[str, Any]
    opt: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value."""
        updates, new_opt_state = self.opt.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step = self.step + 1, params = new_params, opt_state = new_opt_state, **kwargs,)

    def update_learning_rate(self, *, learning_rate):
        """ Updates the learning rate"""
        self.opt_state.hyperparams['learning_rate'] = learning_rate
        return

    def get_optimizer_hparams(self,):
        return self.opt_state.hyperparams

    @classmethod
    def create(cls, *, apply_fn, params, opt, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = opt.init(params)
        return cls(step = 0, apply_fn = apply_fn, params = params, opt = opt, opt_state = opt_state, **kwargs, )

@jax.jit
def compute_accuracy(logits, targets):
    """ Accuracy, used while measuring the state"""
    # Get the label of the one-hot encoded target
    target_class = jnp.argmax(targets, axis = 1)
    # Predict the class of the batch of images using
    predicted_class = jnp.argmax(logits, axis = 1)
    return jnp.mean(predicted_class == target_class)

@partial(jax.jit, static_argnums = 2)
def compute_eval_metrics(state, batch, loss_function):
    imgs, labels = batch
    logits = state.apply_fn({'params': state.params}, imgs)
    # compute loss
    loss = loss_function(logits = logits, labels = labels)
    # compute accuracy using logits
    target_class = jnp.argmax(labels, axis = 1)
    predicted_class = jnp.argmax(logits, axis = 1)
    accuracy = jnp.mean(predicted_class == target_class)
    return loss, accuracy

def compute_eval_metrics_dataset(state, loader, loss_function, num_examples, batch_size):
    """ Description: Estimates the loss and accuracy of a batched data stream """
    ds_loss = 0
    ds_accuracy = 0
    num_batches = estimate_num_batches(num_examples, batch_size)
    for batch_ix in range(num_batches):
        batch = next(loader)
        loss, accuracy = compute_eval_metrics(state, batch, loss_function)
        ds_loss += loss
        ds_accuracy += accuracy
    ds_loss = ds_loss / num_batches
    ds_accuracy = ds_accuracy / num_batches
    return ds_loss, ds_accuracy

@partial(jax.jit, static_argnums = 2)
def loss_step(state: TrainState, batch: Tuple, loss_function):
    "Compute loss for a single batch"
    x, y = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss = loss_function(logits, y)
    return logits, loss

@partial(jax.jit, static_argnums = 2)
def grads_step(state: TrainState, batch: Tuple, loss_function):
    "Compute gradients for a single batch"
    x, y = batch

    def loss_fn(params):
        "loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    return grads, loss

@partial(jax.jit, static_argnums = 2)
def train_step(state: TrainState, batch: Tuple, loss_function):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss

@partial(jax.jit, static_argnums = 2)
def train_sharpness_power_step(state: TrainState, batch: Tuple, loss_function, v, m_iter: int = 20):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    def fori_hvp(i, v):
        return body_hvp(v / norm(v))
    
    v = v / norm(v)
    v = jax.lax.fori_loop(0, m_iter, fori_hvp, v / norm(v))
    sharpness = jnp.vdot(v / norm(v), hvp(flat_params, v / norm(v)))
    
    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss, sharpness, v

@partial(jax.jit, static_argnums = 2)
def train_sharpness_lobpcg_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 100, tol = 1e-09):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        "MSE loss"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    #calculate the gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    #update the state
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss, eigs, eigvs, n_iter

@partial(jax.jit, static_argnums = 2)
def train_pre_sharpness_lobpcg_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 100, tol = 1e-09):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    ### SHARPNESS ESTIMATION

    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    Pdiag = jnp.ones_like(flat_params) # identity for sharpness

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1] / Pdiag

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    ### PRE-SHARPNESS ESTIMATION

    # estimate the diagonal of the pre-conditioner Pdiag
    def compute_Pdiag(vt, beta1, beta2, epsilon, eps_root, t):
        # Compute the diagonal preconditioner matrix
        vhat = vt  / (1 - beta2**(t + 1))
        Pdiag = (jnp.sqrt(vhat + eps_root) + epsilon) * (1 - beta1**(t + 1))
        return Pdiag

    hyperparams = state.opt_state.hyperparams
    beta1 = hyperparams['b1']; beta2 = hyperparams['b2']; epsilon = hyperparams['eps']; eps_root = hyperparams['eps_root']

    inner_state = state.opt_state.inner_state[0] # inner_state[1] is empty, not sure why
    vt = inner_state.nu # https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py
    flat_vt, rebuild_fn = jax.flatten_util.ravel_pytree(vt)

    Pdiag = compute_Pdiag(flat_vt, beta1, beta2, epsilon, eps_root, state.step)

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    pre_eigs, pre_eigvs, pre_n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    ### GRADIENT ESTIMATION
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    ### APPLY GRADIENTS
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss, eigs, eigvs, n_iter, pre_eigs, pre_eigvs, pre_n_iter


@partial(jax.jit, static_argnums = 2)
def train_pre_sharpness_lobpcg_custom_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 100, tol = 1e-09):
    "Compute gradients, loss and accuracy for a single batch"

    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits

    ### SHARPNESS ESTIMATION

    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    Pdiag = jnp.ones_like(flat_params) # identity for sharpness

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1] / Pdiag

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    ### PRE-SHARPNESS ESTIMATION

    # estimate the diagonal of the pre-conditioner Pdiag
    def compute_Pdiag(vt, beta1, beta2, epsilon, t):
        # Compute the diagonal preconditioner matrix
        vhat = vt  / (1 - beta2**(t + 1))
        Pdiag = (jnp.sqrt(vhat) + epsilon) * (1 - beta1**(t + 1))
        return Pdiag

    hyperparams = state.opt_state.hyperparams
    beta1 = hyperparams['b1']; beta2 = hyperparams['b2']; epsilon = hyperparams['eps']

    inner_state = state.opt_state.inner_state # inner_state[1] is empty, not sure why
    vt = inner_state.nu # https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py
    flat_vt, rebuild_fn = jax.flatten_util.ravel_pytree(vt)

    Pdiag = compute_Pdiag(flat_vt, beta1, beta2, epsilon, state.step)

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    pre_eigs, pre_eigvs, pre_n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    ### GRADIENT ESTIMATION
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, logits), grads = grad_fn(state.params)
    ### APPLY GRADIENTS
    state = state.apply_gradients(grads = grads)
    return state, logits, grads, loss, eigs, eigvs, n_iter, pre_eigs, pre_eigvs, pre_n_iter

def compute_sharpness_dataset(state, batches, loss_function, vs, num_batches):
    total_sharpness = 0
    for batch_idx in range(num_batches):
        batch = next(batches)
        eigs, eigvs, n_iter = hessian_lobpcg_step(state, batch, loss_function, vs, m_iter = 1000, tol = 1e-09)
        sharpness = eigs.squeeze()
        total_sharpness += sharpness
    total_sharpness /= num_batches
    return total_sharpness

@partial(jax.jit, static_argnums = 2)
def pre_hessian_lobpcg_custom_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter: int = 100, tol = 1e-09):
    "Compute gradients, loss and accuracy for a single batch"
    x, y = batch

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss

    ### PRE-SHARPNESS ESTIMATION

    # estimate the diagonal of the pre-conditioner Pdiag
    def compute_Pdiag(vt, beta1, beta2, epsilon, t):
        # Compute the diagonal preconditioner matrix
        vhat = vt  / (1 - beta2**(t + 1))
        Pdiag = (jnp.sqrt(vhat) + epsilon) * (1 - beta1**(t + 1))
        return Pdiag

    hyperparams = state.opt_state.hyperparams
    beta1 = hyperparams['b1']; beta2 = hyperparams['b2']; epsilon = hyperparams['eps']

    inner_state = state.opt_state.inner_state # inner_state[1] is empty, not sure why
    vt = inner_state.nu # https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py
    flat_vt, rebuild_fn = jax.flatten_util.ravel_pytree(vt)

    Pdiag = compute_Pdiag(flat_vt, beta1, beta2, epsilon, state.step)

    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1] / Pdiag

    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    pre_eigs, pre_eigvs, pre_n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)

    return pre_eigs, pre_eigvs, pre_n_iter


@partial(jax.jit, static_argnums = 2)
def hessian_lobpcg_step(state: TrainState, batch: Tuple, loss_function, vs, m_iter = 100, tol = 1e-9):
    """
    Description: Compute top-k eigenvalue and hessian using the LOBPCG method
    Inputs:
        -- state: model state for forward pass and parameters
        -- batch: Tuple consisting of input and outputs pairs
        -- loss_function: function, mse or crossentropy
        -- vs: array of initial guesses of the eigenvectors
    Returns:
        -- eigs: top-k eigenvalues
        -- eigvs: top-k eigenvectors
        -- n_iter: number of iterations till convergence
    """
    x, y = batch
    def loss_fn(params):
        "Loss function"
        logits = state.apply_fn({'params': params}, x)
        loss = loss_function(logits, y)
        return loss, logits
    
    flat_params, rebuild_fn = jax.flatten_util.ravel_pytree(state.params)

    def loss_fn_flat(flat_params):
        unflat_params = rebuild_fn(flat_params)
        loss, _ = loss_fn(unflat_params)
        return loss
    
    def hvp(flat_params, v):
        return jax.jvp(jax.grad(loss_fn_flat), [flat_params], [v])[1]
    
    body_hvp = jax.tree_util.Partial(hvp, flat_params)
    body_hvp = jax.vmap(body_hvp, 1, -1)

    vs = vs / norm(vs, axis = -1, keepdims = True)
    eigs, eigvs, n_iter = sparse.linalg.lobpcg_standard(body_hvp, vs, m = m_iter, tol = tol)
    return eigs, eigvs, n_iter

def data_stream(seed, ds, batch_size, augment):
    " Creates a data stream with a predifined batch size."
    train_images, train_labels = ds
    num_train = train_images.shape[0]
    num_batches = estimate_num_batches(num_train, batch_size)
    rng = np.random.RandomState(seed)
    key = jax.random.PRNGKey(seed)

    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1)*batch_size]
            x_batch = train_images[batch_idx]
            y_batch = train_labels[batch_idx]
            if augment:
                key, subkey = jax.random.split(key)
                x_batch, y_batch = data_utils.transform(subkey, (x_batch, y_batch) )
            yield x_batch, y_batch

def estimate_num_batches(num_train, batch_size):
    "Estimates number of batches using dataset and batch size"
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    return num_batches
