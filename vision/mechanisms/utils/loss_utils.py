import optax

def cross_entropy_loss(logits, labels):
  return optax.softmax_cross_entropy(logits = logits, labels = labels).mean()

def cross_entropy_loss_integer_labels(logits, labels):
  return optax.softmax_cross_entropy_with_integer_labels(logits = logits, labels = labels).mean()

def mse_loss(logits, labels):
    """ MSE loss used while measuring the state"""
    return 0.5 * ((logits - labels) ** 2).mean()