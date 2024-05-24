## Adam implementation test

These scripts check the correcteness of custom Adam implementation by comparing it to the optax implementation

Script: 

opt_quad_adam.py comptes the custom Adam implementation of Adam in utils with optax.adam by optimizing over loss function x^2/2

The script takes one argument as argv: the name of the optimizer. 

Valid arguments:

1. adam: uses optax.adam
2. base_adam: uses custom adam implementation from utils.optim_utils.adam_init
