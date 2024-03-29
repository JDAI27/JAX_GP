import jax.numpy as jnp
#import numpy as np
from jax import jacrev,grad, jit, vmap
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

        #kernel_function = jit(kernel_function)
        if kernel_function.__name__ == "_combined_kernel_function":
            self.kernel_function = kernel_function
        else:
            mv = vmap(jit(kernel_function), (0, None, None), 0)
            self.kernel_function = vmap (mv, (None, 0, None), 1)
            #mv = vmap(jit(kernel_function), in_axes=(0, 0, None), out_axes=0)
            #self.kernel_function = vmap(mv, in_axes=(None, 0, None), out_axes=1)
        self.kernel_name = kernel_name
        self.num_params = num_params
        self.param_values = jnp.array(param_values)
        self.param_bounds = param_bounds
        self.eval_gradient = eval_gradient



        if eval_gradient and kernel_function.__name__ != "_combined_kernel_function":
            #self.kernel_gradient = jit(jacfwd(kernel_function,argnums=2))
            #self.kernel_gradient = jit(jacrev(kernel_function,argnums=2))
            g_kernel = jit(grad(kernel_function, argnums=2))
            mv = vmap(g_kernel, (0, None, None), 0)
            self.kernel_gradient = vmap (mv, (None, 0, None), 1)

    def __call__(self, X1, X2 = None):
        """It computes the kernel function with the given input points X1 and X2 and the current kernel parameter values."""
        if X2 == None:
            X2 = X1
        return self.kernel_function(X1, X2, self.param_values)

    def gradient(self, X1, X2, params = None):
        """It computes the gradient of the kernel function with respect to the kernel parameters."""
        if not self.eval_gradient:
            raise ValueError("Gradient evaluation is not enabled for this kernel")
        if params is None:
            params = self.param_values
        return self.kernel_gradient(X1, X2, params)

    def __add__(self, other_kernel):
        return KernelSum(self, other_kernel)

    def __mul__(self, other_kernel):
        return KernelProduct(self, other_kernel)


class KernelOperation(Kernel):
    def __init__(self, kernel1, kernel2):
        # Initialize the two input kernels
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.k1_num = kernel1.num_params
        self.k2_num = kernel2.num_params

        # Calculate the combined number of parameters, parameter values, and parameter bounds
        num_params = kernel1.num_params + kernel2.num_params
        param_values = jnp.append(kernel1.param_values, kernel2.param_values)
        param_bounds = (kernel1.param_bounds) + kernel2.param_bounds
        eval_gradient=True
        
        super().__init__(self._combined_kernel_function, num_params, param_values, param_bounds, eval_gradient,self._combined_kernel_name())

    @partial(jit, static_argnums=(0,))
    def gradient(self, X1, X2,params = None):
        if params is None:
            params = self.param_values
        g1 = self.kernel1.kernel_gradient(X1, X2, params[:self.k1_num])
        g2 = self.kernel2.kernel_gradient(X1, X2, params[self.k1_num:])
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
        return self.kernel1.kernel_function(X1, X2,params[:self.k1_num]) + self.kernel2.kernel_function(X1, X2,params[self.k1_num:])

    @partial(jit, static_argnums=(0,))
    def _combined_kernel_gradient(self, X1, X2, g1, g2):
        return jnp.append(g1, g2,axis=2)
    
    def _combined_kernel_name(self,):
        self.kernel_gradient = jit(lambda X1, X2, params: jnp.append(self.kernel1.kernel_gradient(X1, X2,params[:self.k1_num]),
                                                                self.kernel2.kernel_gradient(X1, X2,params[self.k1_num:]),axis=2))
        return self.kernel1.kernel_name+'+'+self.kernel2.kernel_name


class KernelProduct(KernelOperation):
    @partial(jit, static_argnums=(0,))
    def _combined_kernel_function(self, X1, X2, params):
        return self.kernel1.kernel_function(X1, X2,params[:self.k1_num]) * self.kernel2.kernel_function(X1, X2,params[self.k1_num:])

    @partial(jit, static_argnums=(0,))
    def _combined_kernel_gradient(self, X1, X2, g1, g2):
        return jnp.append(self.kernel1.kernel_function(X1, X2,self.param_values[:self.k1_num])[:,:,jnp.newaxis] * g2, 
                          self.kernel2.kernel_function(X1, X2,self.param_values[self.k1_num:])[:,:,jnp.newaxis] * g1,axis=2)
    
    def _combined_kernel_name(self,):
        self.kernel_gradient = jit(lambda X1, X2, params: jnp.append(self.kernel1.kernel_function(X1, X2,params[:self.k1_num])[:,:,jnp.newaxis] * self.kernel2.kernel_gradient(X1, X2,params[self.k1_num:]), 
                                    self.kernel2.kernel_function(X1, X2,params[self.k1_num:])[:,:,jnp.newaxis] * self.kernel1.kernel_gradient(X1, X2,params[:self.k1_num]),
                                    axis=2))
        return '('+self.kernel1.kernel_name+') * ('+self.kernel2.kernel_name +')'


class ICMKernel(Kernel):
    def __init__(self, base_kernel, rank,output_dim, icm_param_bound = None, **kwargs):
        """
        Initialize the ICM kernel.
        base_kernel: An instance of the Kernel class, which will be used as the base kernel function.
        output_dim: The number of outputs (i.e., the dimensionality of the coregionalization matrix).
        """
        self.base_kernel = base_kernel
        self.output_dim = output_dim
        self.rank = rank

        # Concatenate the coregionalization matrix parameters with the base kernel parameters
        self.param_values = jnp.concatenate([base_kernel.param_values, jnp.ones(rank*output_dim).flatten()])
        if icm_param_bound is None:
            icm_param_bound = [(0, None)] * (rank * output_dim)
        self.param_bounds = base_kernel.param_bounds + icm_param_bound

        super().__init__(kernel_function=self._combined_kernel_function,
                        num_params=len(self.param_values),
                        param_values=self.param_values,
                        param_bounds=self.param_bounds,
                        **kwargs)

    @partial(jit, static_argnums=(0,))
    def _combined_kernel_function(self, X1, X2 = None, params=None):
        """
        Compute the ICM kernel function between input points.

        X, X2: Input points.
        params: Optional parameter values, including the coregionalization matrix.
        """
        if params is None:
            params = self.param_values
        
        if X2 is None:
            X2 = X1

        X1_out = X1[:,-1].flatten().astype(jnp.int32)
        X2_out = X2[:,-1].flatten().astype(jnp.int32)

        base_kernel_params = params[:-self.output_dim * self.rank]
        
        W_params = params[-self.output_dim * self.rank:].reshape(self.output_dim, self.rank)

        compute_B_params = jit(lambda W: jnp.dot(W, W.T)[X1_out[:, None], X2_out])
        B_params = compute_B_params(W_params)

        # Compute the base kernel function
        K_base = self.base_kernel.kernel_function(X1[:,:-1], X2[:,:-1], jnp.array(base_kernel_params))

        #print(K_base.shape)
        #print(B_params.shape)
        # Compute the ICM kernel function
        K_icm = K_base * B_params
        return K_icm

    @partial(jit, static_argnums=(0,))
    def gradient(self, X1, X2 = None,params=None):
        if params is None:
            params = self.param_values

        if X2 is None:
            X2 = X1

        base_kernel_params = params[:-self.output_dim * self.rank]
        W_params = params[-self.output_dim * self.rank:].reshape(self.output_dim, self.rank)

        B_params = jnp.dot(W_params, W_params.T)
        K_base = self.base_kernel.kernel_function(X1[:,:-1], X2[:,:-1], base_kernel_params)
        K_base_gradient = self.base_kernel.kernel_gradient(X1[:,:-1], X2[:,:-1], base_kernel_params)

        # Compute the gradient of the ICM kernel function
        X1_out = X1[:,-1].flatten().astype(jnp.int32)
        X2_out = X2[:,-1].flatten().astype(jnp.int32)
        compute_B_params = jit(lambda W: jnp.dot(W, W.T)[X1_out[:, None], X2_out])
        B_params = compute_B_params(W_params)
        gradient_W_B = jit(jacrev(compute_B_params, argnums=0))
        B_params_gradient = gradient_W_B(W_params).reshape(X1.shape[0], X2.shape[0], self.output_dim*self.rank)
        #print("Shape of K_base:", K_base.shape)
        #print("Shape of K_base_gradient:", K_base_gradient.shape)
        #print("Shape of B_params:", B_params.shape)
        #print("Shape of B_params_gradient:", B_params_gradient.shape)
        g1 = g1 = (K_base_gradient * B_params[..., None])
        g2 = (K_base[..., None] * B_params_gradient)
        return jnp.append(g1, g2,axis=2)
        # Implementation depends on the desired gradient structure
        #raise NotImplementedError("Gradient computation not implemented for ICM kernel")


def rbf_kernel(X1, X2, params):
    params = jnp.array(params)
    if len(params) == 1:
        # isotropic RBF kernel with length scale = params[0]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        return jnp.exp(-0.5 * sq_norm / params[0] ** 2)
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1 - X2) ** 2
        return jnp.exp(-0.5 * jnp.sum(sq_diff / params ** 2, axis=-1))
    else:
        raise ValueError("Incorrect number of parameters for RBF kernel")
        

def matern12(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return jnp.exp(-norm / params[0])
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1 - X2) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return jnp.exp(-norm)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")


def matern32(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return (1.0 + jnp.sqrt(3.0) * norm / params[0]) * jnp.exp(-jnp.sqrt(3.0) * norm / params[0])
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1 - X2) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return (1.0 + jnp.sqrt(3.0) * norm) * jnp.exp(-jnp.sqrt(3.0) * norm)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")
    
def matern52(X1, X2, params):
    if len(params) == 1:
        # isotropic Matern kernel with length scale = params[0]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        inner = jnp.sqrt(5.0) * norm / params[0]
        return (1.0 + inner + 5.0 / 3.0 * sq_norm / (params[0] ** 2)) * jnp.exp(-inner)
    elif len(params) == X1.shape[-1]:
        # ARD (automatic relevance determination) kernel with length scales = params
        sq_diff = (X1 - X2) ** 2
        sq_diff /= params ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        inner = jnp.sqrt(5.0) * norm
        return (1.0 + inner + 5.0 / 3.0 * jnp.sum(sq_diff, axis=-1)) * jnp.exp(-inner)
    else:
        raise ValueError("Incorrect number of parameters for Matern kernel")
    
def exp_sine_squared(X1, X2, params):
    if len(params) == 2:
        # isotropic ExpSineSquared kernel with length scale = params[0] and period = params[1]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        norm = jnp.sqrt(sq_norm)
        return jnp.exp(-2 * jnp.sin(jnp.pi * norm / params[1]) ** 2 / params[0] ** 2)
    elif len(params) == X1.shape[-1] + 1:
        # ARD ExpSineSquared kernel with length scales = params[:-1] and period = params[-1]
        sq_diff = (X1 - X2) ** 2
        sq_diff /= params[:-1] ** 2
        norm = jnp.sqrt(jnp.sum(sq_diff, axis=-1))
        return jnp.exp(-2 * jnp.sin(jnp.pi * norm / params[-1]) ** 2 / jnp.sum(params[:-1] ** 2))
    else:
        raise ValueError("Incorrect number of parameters for ExpSineSquared kernel")

def rational_quadratic(X1, X2, params):
    if len(params) == 2:
        # isotropic RationalQuadratic kernel with length scale = params[0] and alpha = params[1]
        sq_norm = jnp.sum((X1 - X2) ** 2, axis=-1)
        return (1 + sq_norm / (2 * params[1] * params[0] ** 2)) ** (-params[1])
    elif len(params) == X1.shape[-1] + 1:
        # ARD RationalQuadratic kernel with length scales = params[:-1] and alpha = params[-1]
        sq_diff = (X1 - X2) ** 2
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
