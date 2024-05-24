import jax.numpy as jnp
import jax
import pickle as pl
import tensorflow_datasets as tfds

def _one_hot(x, k, dtype = jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def _standardize(x, abc = 'ntp'):
  """Standardization per sample across feature dimension."""
  axes = tuple(range(1, len(x.shape)))  
  mean = jnp.mean(x, axis = axes, keepdims = True)
  std_dev = jnp.std(x, axis = axes, keepdims = True)
  normx = (x - mean) / std_dev
  if abc == 'mup':
    in_dim = jnp.prod(jnp.array(x.shape[1:]))
    normx /= jnp.sqrt(in_dim)
  return normx

def _random_crop(key, x, pixels):
    """x should have shape [batch, img_dim, img_dim, channels]."""
    zero = (0, 0)
    pixpad = (pixels, pixels)
    paddedX = jnp.pad(x, (zero, pixpad, pixpad, zero), 'reflect')
    corner = jax.random.randint(key, (x.shape[0], 2), 0, 2 * pixels)
    assert x.shape[1] == x.shape[2]
    img_size = x.shape[1]
    slices = [(slice(int(o[0]), int(o[0]) + img_size), slice(int(o[1]), int(o[1]) + img_size), slice(None)) for x, o in zip(paddedX, corner)]
    paddedX = jnp.concatenate([x[jnp.newaxis, s[0], s[1], s[2]] for x, s in zip(paddedX, slices)])
    return paddedX

def _random_horizontal_flip(key, x, prob):
    """Perform horizontal flip with probability prob"""
    assert x.shape[1] == x.shape[2] # check wheather its a square image
    flip = jax.random.uniform(key, shape = (len(x), 1, 1, 1))
    flippedX = x[:, :, ::-1, :]
    x = jnp.where(flip < prob, flippedX, x)
    return x

def _random_vertical_flip(key, x, prob):
    """Perform vertical flip along axis with probability prob"""
    assert x.shape[1] == x.shape[2] # check wheather its a square image
    flip = jax.random.uniform(key, shape = (len(x), 1, 1, 1))
    flippedX = x[:, ::-1, :, :]
    x = jnp.where(flip < prob, flippedX, x)
    return x

"currently this function does not for generic image sizes"

def crop(key, batch):
    """Random crops."""
    image, label = batch
    img_size = image.shape

    pixels = 4 #
    pixpad = (pixels, pixels)
    zero = (0, 0)
    padded_image = jnp.pad(image, (pixpad, pixpad, zero))
    corner = jax.random.randint(key, (2,), 0, 2 * pixels)
    corner = jnp.concatenate((corner, jnp.zeros((1,), jnp.int32)))
    cropped_image = jax.lax.dynamic_slice(padded_image, corner, img_size)
    return cropped_image, label

batched_crop = jax.vmap(crop, 0, 0)

def mixup(key, batch):

    """
    Mixup data augmentation: Mixes two training examples with weight from beta distribution
                            for alpha = 1.0, it draws from uniform distribution

    """

    image, label = batch

    N = image.shape[0]

    #weight = jax.random.beta(key, alpha, alpha, (N, 1)) This was causing issues with jitting; dont know why. It works well
    weight = jax.random.uniform(key, (N, 1))
    mixed_label = weight * label + (1.0 - weight) * label[::-1, :]

    weight = jnp.reshape(weight, (N, 1, 1, 1))
    mixed_image = weight * image + (1.0 - weight) * image[::-1, :, :, :]

    return mixed_image, mixed_label

@jax.jit
def transform(key, batch):
    """Apply horizontal flip, crop, and mixup transformations to a batch"""
    
    imgs, labels = batch
    num_imgs = imgs.shape[0]
    key1, key2, key3 = jax.random.split(key, 3)
    
    # use key 1 to perform random horizontal flip
    imgs = _random_horizontal_flip(key, imgs, prob = 0.5) # the last argument is the probability
    
    # use key2 for cropping
    batch_split = jax.random.split(key2, num_imgs)
    imgs, labels = batched_crop(batch_split, (imgs, labels))

    #use key 3 for mixup
    imgs, labels = mixup(key3, (imgs, labels))
    return imgs, labels

def load_image_data(dir: str, ds_name: str, flatten: bool = True, subset = False, num_examples: int = 1000):
    """
    Description: loads existing dataset from a directory

    Inputs: 
      1. dir: directory where the data is saved
      2. dataset: dataset name, the existing file should be dataset.dump
      3. num_examples: num_examples required
    """

    train_path = f'{dir}/{ds_name}/{ds_name}.train'    
    with open(train_path, 'rb') as fi:
      (x_train, y_train) = pl.load(fi)

    test_path = f'{dir}/{ds_name}/{ds_name}.test'    
    with open(test_path, 'rb') as fi:
      (x_test, y_test) = pl.load(fi)

    # Flatten the image for FCNs
    if flatten:
      x_train = x_train.reshape((x_train.shape[0], -1))
      x_test = x_test.reshape((x_test.shape[0], -1))
    
    # consider a subset of the existing dataset
    if subset:
    
      x_train = x_train[:num_examples]
      y_train = y_train[:num_examples]
    
      x_test = x_test[:num_examples]
      y_test = y_test[:num_examples]
    
    # move the dataset to the GPU memory
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)

    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)

    return (x_train, y_train), (x_test, y_test)


def load_image_data_tfds(dataset: str, flatten: bool = True, subset = True, num_examples: int = 1000):
    """
    Description: loads existing dataset from a directory

    Inputs: 
      1. dir: directory where the data is saved
      2. dataset: dataset name, the existing file should be dataset.dump
      3. num_examples: num_examples required
    """
    if dataset in ['cifar100', 'cifar10', 'mnist', 'fashion_mnist']:
        ds_train, ds_test = tfds.as_numpy(tfds.load(dataset, data_dir='./', split=["train", "test"], batch_size=-1, as_dataset_kwargs={"shuffle_files": False}))
    else:
        raise ValueError("Invalid dataset name.")

    x_train, y_train, x_test, y_test = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])

    x_train = jnp.asarray(x_train, dtype = jnp.float32)
    y_train = jnp.asarray(y_train, dtype = jnp.int32)

    x_test = jnp.asarray(x_test, dtype = jnp.float32)
    y_test = jnp.asarray(y_test, dtype = jnp.int32)


    # Flatten the image for FCNs
    if flatten:
      x_train = x_train.reshape((x_train.shape[0], -1))
      x_test = x_test.reshape((x_test.shape[0], -1))
    
    # consider a subset of the existing dataset
    if subset:
      x_train = x_train[:num_examples]
      y_train = y_train[:num_examples]
    
      x_test = x_test[:num_examples]
      y_test = y_test[:num_examples]
      
    # move the dataset to the GPU memory
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)

    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)

    return (x_train, y_train), (x_test, y_test)
