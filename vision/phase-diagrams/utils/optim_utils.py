from typing import NamedTuple
import jax.flatten_util
import jax.numpy as jnp
import optax
import jax
import flax

class AdamState(NamedTuple):
    mu: optax.Updates  # Moving average of the gradients
    nu: optax.Updates  # Moving average of the squared gradients
    count: jnp.ndarray  # Timestep


def adam_init(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, grads = None):

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        if grads is not None:
            nu = jax.tree_map(lambda x: x**2, grads)  
        # key = jax.random.PRNGKey(0)
        #nu = jax.tree_map(lambda x: jax.random.normal(key, shape = x.shape, dtype = x.dtype), params)
        return AdamState(mu = mu, nu = nu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, b1 = b1, b2 = b2, eps = eps):
        mu, nu, count = state.mu, state.nu, state.count + 1
        # flatten everything
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_mu, rebuild_fn = jax.flatten_util.ravel_pytree(mu)
        flat_nu, rebuild_fn = jax.flatten_util.ravel_pytree(nu)

        flat_mu_next = b1 * flat_mu + (1 - b1) * flat_grads
        flat_mu_hat = flat_mu_next / (1 - b1 ** count)

        flat_nu_next = b2 * flat_nu + (1 - b2) * (flat_grads**2)
        flat_nu_hat = flat_nu_next / (1 - b2 ** count)

        flat_updates =  -learning_rate * flat_mu_hat / (jnp.sqrt(flat_nu_hat) + eps)
            
        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)
        nu_next = rebuild_fn(flat_nu_next)

        return updates, AdamState(mu = mu_next, nu = nu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)


def adam_mup_init(lr_pytree, learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, grads = None):

    def init_fn(params):        
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        if grads is not None:
            nu = jax.tree_map(lambda x: x**2, grads) # 
        # key = jax.random.PRNGKey(0)
        #nu = jax.tree_map(lambda x: jax.random.normal(key, shape = x.shape, dtype = x.dtype), params)
        return AdamState(mu = mu, nu = nu, count = jnp.zeros([]))

    def update_fn(grads, state, params = None, learning_rate = learning_rate, b1 = b1, b2 = b2, eps = eps):
        
        mu, nu, count = state.mu, state.nu, state.count + 1
        # flatten grads, mu, nu
        flat_grads, rebuild_fn = jax.flatten_util.ravel_pytree(grads)
        flat_mu, rebuild_fn = jax.flatten_util.ravel_pytree(mu)
        flat_nu, rebuild_fn = jax.flatten_util.ravel_pytree(nu)

        # update mu
        flat_mu_next = b1 * flat_mu + (1 - b1) * flat_grads
        flat_mu_hat = flat_mu_next / (1 - b1 ** count)
        
        # update new
        flat_nu_next = b2 * flat_nu + (1 - b2) * (flat_grads**2)
        flat_nu_hat = flat_nu_next / (1 - b2 ** count)

        # updates except the learning rate
        flat_updates =  -flat_mu_hat / (jnp.sqrt(flat_nu_hat) + eps)
        
        # unflatten back
        updates = rebuild_fn(flat_updates)
        mu_next = rebuild_fn(flat_mu_next)
        nu_next = rebuild_fn(flat_nu_next)

        # estimate the learning rate dictionary

        # multiply with learning rate
        updates = jax.tree_map(lambda lr,g: learning_rate*lr*g, lr_pytree, updates)

        return updates, AdamState(mu = mu_next, nu = nu_next, count = count)

    return optax.GradientTransformation(init_fn, update_fn)



def flatten_pytree(pytree, prefix = ''):
    # This function extracts variance for each layer and organizes it into a readable format
    flat_dict = {}
    for key, value in pytree.items():
        # Construct the new key path
        new_key = f'{prefix}.{key}' if prefix else key

        if isinstance(value, dict):
            # If the value is a dictionary, recurse further
            flat_dict.update(flatten_pytree(value, new_key))
        else:
            # Otherwise, store the value with its accumulated key path
            flat_dict[new_key] = value
    return flat_dict


