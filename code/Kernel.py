import jax.numpy as jnp
#import numpy as np
from jax import jacfwd, jit
from functools import partial


class Kernel:
    def __init__(self, kernel_function, num_params, param_values, param_bounds, eval_gradient=True,kernel_name=" "):
        '''
        This is the constructor method for the Kernel class. 

            kernel_function: The kernel function, which computes the covariance between input points.

            kernel_name: A string representing the name of the kernel.

            num_params: The number of parameters required for this kernel function.

            param_values: A JAX array containing the initial values of the kernel parameters.

            param_bounds: A list of tuples specifying the bounds for each kernel parameter.

            eval_gradient: A boolean flag indicating whether gradient evaluation should be enabled for this kernel.

            If eval_gradient is True, the kernel_gradient attribute is initialized using the JAX jacfwd function, 
            which computes the forward-mode Jacobian of the kernel_function with respect to the kernel parameters.
        '''

        self.kernel_function = kernel_function
        self.kernel_name = kernel_name
        self.num_params = num_params
        self.param_values = jnp.array(param_values)
        self.param_bounds = param_bounds
        self.eval_gradient = eval_gradient

        if eval_gradient:
            self.kernel_gradient = jit(jacfwd(kernel_function,argnums=2))

    def __call__(self, X1, X2):
        """It computes the kernel function with the given input points X1 and X2 and the current kernel parameter values."""
        return self.kernel_function(X1, X2, self.param_values)

    def gradient(self, X1, X2):
        if not self.eval_gradient:
            raise ValueError("Gradient evaluation is not enabled for this kernel")
        return self.kernel_gradient(X1, X2, self.param_values)

    def __add__(self, other_kernel):
        return KernelSum(self, other_kernel)

    def __mul__(self, other_kernel):
        return KernelProduct(self, other_kernel)


class KernelOperation(Kernel):
    def __init__(self, kernel1, kernel2):
        # Initialize the two input kernels
        self.kernel1 = kernel1
        self.kernel2 = kernel2

        # Calculate the combined number of parameters, parameter values, and parameter bounds
        num_params = kernel1.num_params + kernel2.num_params
        param_values = jnp.append(kernel1.param_values, kernel2.param_values)
        param_bounds = (kernel1.param_bounds) + kernel2.param_bounds
        eval_gradient=True
        
        super().__init__(self._combined_kernel_function, num_params, param_values, param_bounds, eval_gradient,self._combined_kernel_name())

    @partial(jit, static_argnums=(0,))
    def gradient(self, X1, X2):
        g1 = self.kernel1.gradient(X1, X2)
        g2 = self.kernel2.gradient(X1, X2)
        return self._combined_kernel_gradient(X1, X2, g1, g2)

    def _combined_kernel_function(self, X1, X2, params):
        raise NotImplementedError("Subclasses should implement this method")

    def _combined_kernel_gradient(self, X1, X2, g1, g2):
        raise NotImplementedError("Subclasses should implement this method")
    def _combined_kernel_name(self,):
        raise NotImplementedError("Subclasses should implement this method")


class KernelSum(KernelOperation):
    @partial(jit, static_argnums=(0,))
    def _combined_kernel_function(self, X1, X2, params):
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)

    @partial(jit, static_argnums=(0,))
    def _combined_kernel_gradient(self, X1, X2, g1, g2):
        return jnp.append(g1, g2,axis=2)
    
    def _combined_kernel_name(self,):
        return self.kernel1.kernel_name+'+'+self.kernel2.kernel_name


class KernelProduct(KernelOperation):
    @partial(jit, static_argnums=(0,))
    def _combined_kernel_function(self, X1, X2, params):
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)

    @partial(jit, static_argnums=(0,))
    def _combined_kernel_gradient(self, X1, X2, g1, g2):
        return jnp.append(self.kernel1(X1, X2)[:,:, None] * g2, self.kernel2(X1, X2)[:,:, None] * g1,axis=2)
    
    def _combined_kernel_name(self,):
        return self.kernel1.kernel_name+'* ('+self.kernel2.kernel_name +" )"


def rbf_kernel(X1, X2, params):
    params = jnp.array(params)
    if len(params) == 1:
        # isotropic RBF kernel with length scale = params[0]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        return jnp.exp(-0.5 * sq_norm / params[0] ** 2)
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        return jnp.exp(-0.5 * jnp.sum(sq_diff / params ** 2, axis=-1))
    else:
        raise ValueError("Incorrect number of parameters for RBF kernel")
        

def matern12(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return jnp.exp(-norm / params[0])
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return jnp.exp(-norm)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")


def matern32(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return (1.0 + jnp.sqrt(3.0) * norm / params[0]) * jnp.exp(-jnp.sqrt(3.0) * norm / params[0])
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return (1.0 + jnp.sqrt(3.0) * norm) * jnp.exp(-jnp.sqrt(3.0) * norm)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")
    
def matern52(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        inner = jnp.sqrt(5.0) * norm / params[0]
        return (1.0 + inner + 5.0 / 3.0 * sq_norm / (params[0] ** 2)) * jnp.exp(-inner)
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        inner = jnp.sqrt(5.0) * norm
        return (1.0 + inner + 5.0 / 3.0 * jnp.sum(sq_diff, axis=-1)) * jnp.exp(-inner)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")
    
def exp_sine_squared(X1, X2, params):
    if len(params) == 2:
        # isotropic ExpSineSquared kernel with length scale = params[0] and period = params[1]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return jnp.exp(-2 * jnp.sin(jnp.pi * norm / params[1]) ** 2 / params[0] ** 2)
    elif len(params) == X1.shape[-1] + 1:
        # ARD ExpSineSquared kernel with length scales = params[:-1] and period = params[-1]
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        sq_diff /= params[:-1] ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return jnp.exp(-2 * jnp.sin(jnp.pi * norm / params[-1]) ** 2 / jnp.sum(params[:-1] ** 2))
    else:
        raise ValueError("Incorrect number of parameters for ExpSineSquared kernel")

def rational_quadratic(X1, X2, params):
    if len(params) == 2:
        # isotropic RationalQuadratic kernel with length scale = params[0] and alpha = params[1]
        sq_norm = jnp.sum((X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2, axis=-1)
        return (1 + sq_norm / (2 * params[1] * params[0] ** 2)) ** (-params[1])
    elif len(params) == X1.shape[-1] + 1:
        # ARD RationalQuadratic kernel with length scales = params[:-1] and alpha = params[-1]
        sq_diff = (X1[:, jnp.newaxis, :] - X2[jnp.newaxis, :, :]) ** 2
        sq_diff /= params[:-1] ** 2
        sq_norm = jnp.sum(sq_diff, axis=-1)
        return (1 + sq_norm / (2 * params[-1] * jnp.sum(params[:-1] ** 2))) ** (-params[-1])
    else:
        raise ValueError("Incorrect number of parameters for RationalQuadratic kernel")

def polynomial_kernel(X1, X2, params):
    if len(params) == 2:
        # isotropic polynomial kernel with scale = params[0] and degree = params[1]
        scale = params[0]
        degree = params[1]
        dot_product = jnp.dot(X1, X2.T)
        return ( dot_product / scale**2 + 1) ** degree
    elif len(params) == X1.shape[-1] + 1:
        # ARD polynomial kernel with scales = params[:-1] and degree = params[-1]
        scales = params[:-1]
        degree = params[-1]
        scaled_X1 = X1 / scales
        scaled_X2 = X2 / scales
        dot_product = jnp.dot(scaled_X1, scaled_X2.T)
        return (dot_product + 1) ** degree
    else:
        raise ValueError("Incorrect number of parameters for Polynomial kernel")
