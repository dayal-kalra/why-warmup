import jax.numpy as jnp

def polynomial_warmup(step, init_value, end_value, warm_steps, exponent = 1.0):
    "lr = lr_max * step / step_max"
    warmup_rate = step / warm_steps
    lr = init_value + (end_value - init_value) * warmup_rate ** exponent
    return min(end_value, lr)

def cosine_decay_schedule(step, init_value, end_value, cosine_steps, exponent = 1.0):
    cosine_decay = 0.5 * (1 + jnp.cos( jnp.pi * step / cosine_steps))
    return end_value + (init_value - end_value) * cosine_decay ** exponent