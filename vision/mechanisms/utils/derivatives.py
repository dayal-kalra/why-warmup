import jax
from jax import numpy as jnp, jacrev, jacfwd, jvp
from jax.tree_util import tree_leaves


def smart_jacobian(f, argnums = 0, has_aux = False, return_value = False):
    """

    Description: computes the Jacobian of a function f in a smart way, choosing between reverse-mode or forward-mode automatic differentiation based on the dimensions of the inputs and outputs.

    Inputs: 
        f: function 
        argnums: the arguments wrt the jacobian is calculated 
        has_aux: wheather the function returns any auxilary information like logits
        return_value: wheather to return the function value along with the jacobian
    Output:
        jacobian function of the f which may return the function with output

    """
    
    def jacfun(*args, **kwargs):
        # modify the function 'f' to also return its output evaluated at the arguments alongside its jacobian
        if return_value:
            # the modified function _f
            def _f(*args, **kwargs):
                # the function always returns a tuple for consistency
                if has_aux:
                    out, aux = f(*args, **kwargs)
                    return out, (out, aux)
                else:
                    out = f(*args, **kwargs)
                    return out, out

            _jacfun = smart_jacobian(_f, argnums = argnums, has_aux = True, return_value = False)

            jac, aux = _jacfun(*args, **kwargs)

            if has_aux:
                out, *aux = aux
                return out, jac, aux
            else:
                out = aux
                return out, jac

        inputs = (args[argnums] if isinstance(argnums, int) else [args[i] for i in argnums])
        
        # evaluate shape of a function given inputs without using flops

        if has_aux:
            out_shape = jax.eval_shape(f, *args, **kwargs)[0] # if the function has aux, use the first function output
        else:
            out_shape = jax.eval_shape(f, *args, **kwargs)

        if isinstance(out_shape, jax.ShapeDtypeStruct) and out_shape.shape == ():
            # if f is a scalar, return the grad
            return jax.grad(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
        else:
            # if vector/matrix valued, compute input output dimension
            in_dim = sum(jnp.size(leaf) for leaf in tree_leaves(inputs))
            out_dim = sum(jnp.size(leaf) for leaf in tree_leaves(out_shape))
            # if input dim is greater use jacrev (backprop)
            if in_dim >= out_dim:
                return jacrev(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
            #else, use jacfwd and calculate gradients during forward pass
            else:
                return jacfwd(f, argnums=argnums, has_aux=has_aux)(*args, **kwargs)
    return jacfun



def D(f, x, order = 1, *vs, return_all = False):

    """
    Description: nth order Differential operator D^order f, contracted along given set of vectors vs
    
    Inputs:
        f: function to be differentiated
        x: input
        order: order of differentiation
        vs:  Variable-length argument list representing the directions for contraction
        return_all: If True, returns all intermediate derivatives up to the specified order.

    """

    if return_all:

        def _f(x):
            out = f(x)
            return out, (out,)

        return _D(_f, x, order, *vs, return_all = return_all)[1]
    else:
        _f = f
        return _D(_f, x, order, *vs, return_all = return_all)


def _D(f, x, order=1, *vs, return_all=False):

    assert len(vs) <= order # check if the number of vectors for projection are smaller than the order

    if order == 0:
        return f(x) # no derivatives
    
    elif order == len(vs):
        v, *vs = vs # if order is equal to len(vs) then separate the first item from the list as we only require order-1 vs
    else:
        v = None

    def Df(x):
        if return_all:
            if v is None:
                jac, hist = smart_jacobian(f, has_aux=True)(x)
                return jac, (*hist, jac)
            else:
                _, jac, hist = jvp(f, [x], [v], has_aux=True)
                return jac, (*hist, jac)
        else:
            if v is None:
                return smart_jacobian(f)(x)
            else:
                return jvp(f, [x], [v])[1]

    return _D(Df, x, order - 1, *vs, return_all=return_all)