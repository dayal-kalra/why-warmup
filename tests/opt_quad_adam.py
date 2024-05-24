import jax
import jax.numpy as jnp
import optax
import utils.optim_utils as optim
from sys import argv

# Define the loss function
def loss_fn(x):
    return x**2 / 2

# Initialize the parameter
x = jnp.array(1.0)

opt_name = argv[1]
# Create an optimizer
if opt_name == 'adam':
    opt = optax.adam(learning_rate = 1e-03)
elif opt_name == 'base_adam': # custom implementation of adam
    opt = optim.adam_init(learning_rate = 1e-03)
elif opt_name == 'grads_adam':
    grads_init = jax.grad(loss_fn)(x)
    opt = optim.adam_init(learning_rate = 1e-03, grads = grads_init)
else:
    print(f'Optimizer not supported')
    exit(0)
# Initialize optimizer state
opt_state = opt.init(x)

# Update function
@jax.jit
def update(x, opt_state):
    # Compute the gradient
    grads = jax.grad(loss_fn)(x)
    # Update parameters and optimizer state
    updates, opt_state = opt.update(grads, opt_state, x)
    x = optax.apply_updates(x, updates)
    return x, opt_state

# Optimization loop
for i in range(100):
    x, opt_state = update(x, opt_state)
    print(f"Step {i+1}, x: {x:0.6f}, Loss: {loss_fn(x):0.6f}")

