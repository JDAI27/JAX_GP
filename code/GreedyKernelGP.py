import jax.numpy as jnp
from jax.config import config

from kernel_jax import *
from kernel_jax import Kernel
from gp_jax import GaussianProcess
config.update('jax_enable_x64', True)

matern12 = jit(matern12)
matern32 = jit(matern32)
matern52 = jit(matern52)
rbf_kernel = jit(rbf_kernel)
rational_quadratic = jit(rational_quadratic)
exp_sine_squared = jit(exp_sine_squared)
polynomial_kernel = jit(polynomial_kernel)


class GreedyKernelSearchGP(GaussianProcess):
    def __init__(self, layers: int, sigma_n=1e-6, optimizer='adam', **optimizer_kwargs):
        self.layers = layers
        self.base_kernel = None
        self.kernels = None
        self.sigma_n = sigma_n
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.combinations = ['add', 'multiply']

    def bic(self, lml, k):
        n = len(self.X_train)
        return -2 * lml + k * jnp.log(n)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        Num_length_scale = X.shape[-1]

        if self.kernels is None:
            self.kernels = [Kernel(rbf_kernel,Num_length_scale, jnp.ones(Num_length_scale), [(1e-5, 1e5)] * Num_length_scale, kernel_name='rbf'),
                            Kernel(matern12,Num_length_scale, jnp.ones(Num_length_scale), [(1e-5, 1e5)] * Num_length_scale, kernel_name='matern12'),
                            Kernel(matern32,Num_length_scale, jnp.ones(Num_length_scale), [(1e-5, 1e5)] * Num_length_scale, kernel_name='matern32'), 
                            Kernel(matern52,Num_length_scale, jnp.ones(Num_length_scale), [(1e-5, 1e5)] * Num_length_scale, kernel_name='matern52'),
                            Kernel(rational_quadratic,Num_length_scale+1, jnp.ones(Num_length_scale+1), [(1e-5, 1e5)] * (Num_length_scale+1), kernel_name='rational_quadratic'),
                            Kernel(exp_sine_squared,Num_length_scale+1, jnp.ones(Num_length_scale+1), [(1e-5, 1e5)] * (Num_length_scale+1), kernel_name='exp_sine_squared'),
                            Kernel(polynomial_kernel,Num_length_scale+1, jnp.ones(Num_length_scale+1), [(1e-5, 1e5)] * (Num_length_scale+2), kernel_name='polynomial_kernel')]
        

        for layer in range(self.layers):
            best_bic = jnp.inf
            best_kernel = None

            for combination in self.combinations:
                for kernel in self.kernels:
                    if self.base_kernel is None:
                        combined_kernel = kernel
                    elif combination == 'add':
                        combined_kernel = self.base_kernel + kernel
                    else:
                        combined_kernel = self.base_kernel * kernel

                    print(layer," : ", combined_kernel.kernel_name)

                    _ = GaussianProcess(kernel=combined_kernel, sigma_n=self.sigma_n, optimizer=self.optimizer, **self.optimizer_kwargs)
                    _.fit(X, y)

                    lml = _.LML_
                    current_bic = self.bic(lml, combined_kernel.num_params)
                    
                    print(" with parameters: ", _.kernel.param_values)
                    print(" with bic: ", current_bic)

                    if current_bic < best_bic:
                        best_bic = current_bic
                        best_kernel = combined_kernel
                        print("!!!! best kernel: ", best_kernel.kernel_name, " with bic: ", best_bic)
            print("At", layer, " :  with finial choose of", best_kernel.kernel_name," with bic: ", best_bic)

            self.base_kernel = best_kernel
        super().__init__(self.base_kernel, self.sigma_n, self.optimizer, **self.optimizer_kwargs)

        super().fit(X, y)



    def predict(self, X_star, return_cov=False):
        self.kernel = self.base_kernel
        return super().predict(X_star, return_cov)
    
