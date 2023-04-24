#%%
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad,random
from functools import partial
from jax.scipy.linalg import cho_factor, cho_solve
from jax.example_libraries import optimizers
from jax.scipy.optimize import minimize
from jax.scipy.special import expit as logistic
from jax.scipy.stats import norm
config.update('jax_enable_x64', True)


#%%
class GaussianProcess:
    def __init__(self, kernel=None, sigma_n=1e-5, optimizer='adam', **optimizer_kwargs):
        self.kernel = kernel  # Kernel function for the Gaussian Process
        self.sigma_n = sigma_n  # Noise level (variance) for the observations
        self.optimizer = optimizer  # Optimizer for kernel parameter optimization
        self.optimizer_kwargs = optimizer_kwargs  # Additional arguments for the optimizer
        self.X_train = None  # Training data (features)
        self.y_train = None  # Training data (targets)
        self.alpha = None  # Alpha vector for making predictions
        self.K = None  # Covariance matrix of the training data
        self.LML_ = None

        # Set the optimizer function based on the optimizer name
        if self.optimizer == 'adam':
            self.optimizer = self.adam_optimizer
        elif self.optimizer == 'scipy':
            self.optimizer = self.scipy_minimize
        
        # If the kernel is gradient-based, create the gradient function for Log Marginal Likelihood
        if self.kernel.eval_gradient:
            self._lml_grad = jit(grad(self._log_marginal_likelihood))

    # Incomplete function for optimizing kernel parameters using Scipy's BFGS optimizer
    def scipy_minimize(self, obj_func, initial_theta, bounds):
        res = minimize(obj_func, initial_theta, method='BFGS')
        theta_opt = res.x
        bounds = jnp.array(bounds)
        theta_opt = jnp.array([jnp.clip(param, bounds[i, 0], bounds[i, 1]) for i, param in enumerate(theta_opt)])
        func_min = obj_func(theta_opt)[0]
        return theta_opt, func_min

    # Function for optimizing kernel parameters using JAX's Adam optimizer
    def adam_optimizer(self, obj_func, initial_theta, bounds, num_epochs=1000, step_size=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size, b1=beta1, b2=beta2, eps=eps)
        state = opt_init(initial_theta)
        bounds = jnp.atleast_2d(bounds)
        #bounds = jnp.log(jnp.array(bounds))

        # Define a single optimization step
        @jit
        def step(i, opt_state):
            params = get_params(opt_state)
            value, grads = obj_func(params)

            # Clip the gradients according to the bounds
            clipped_grads = jnp.array([jnp.where((params[i] + grads[i] > bounds[i, 0]) & (params[i] + grads[i] < bounds[i, 1]), grads[i], 0)
                                        for i in range(len(grads))])

            updated_state = opt_update(i, clipped_grads, opt_state)
            return updated_state, value

        # Perform optimization
        for i in range(num_epochs):
            state, loss = step(i, state)

        theta_opt = get_params(state)
        func_min = obj_func(theta_opt)[0]
        return theta_opt, func_min

    # Function for fitting the Gaussian Process model to the given data
    def fit(self, X, y):
        self.X_train = jnp.atleast_2d(X)  # Store the training data (inputs)
        self.y_train = jnp.atleast_2d(y).T  # Store the training data (targets)

        # Optimize kernel parameters
        self._optimize_kernel_params()

        # Compute alpha with optimized kernel parameters
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * jnp.eye(len(self.X_train))
        L = jnp.linalg.cholesky(self.K)
        self.alpha = cho_solve((L, True), self.y_train)

    # Compute the log-marginal likelihood given kernel parameters
    @partial(jit, static_argnums=(0,))
    def _log_marginal_likelihood(self, params):
        X, y = self.X_train, self.y_train
        #print(params.shape)
        #print(self.kernel.kernel_function(X, X,params).shape)
        K = self.kernel.kernel_function(X, X,params)+ self.sigma_n**2 * jnp.eye(len(X))
        try:
            L = jnp.linalg.cholesky(K)
        except jnp.linalg.LinAlgError:
            return (-jnp.inf, jnp.zeros_like(self.kernel.param_values)) if self.kernel.eval_gradient else -jnp.inf
        alpha = cho_solve((L, True), y)
        log_likelihood = -0.5 * jnp.dot(y.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - X.shape[0] / 2 * jnp.log(2 * jnp.pi)
        return log_likelihood

    # Compute the loss and gradients of the log-marginal likelihood given kernel parameters
    @partial(jit, static_argnums=(0,))
    def _loss_and_grads(self, params):
        # Compute the covariance matrix (K) using the kernel function and add the noise term
        K = self.kernel.kernel_function(self.X_train, self.X_train, params) + self.sigma_n**2 * jnp.eye(self.X_train.shape[0])

        # Compute the Cholesky decomposition (L) of the covariance matrix (K).
        L = jnp.linalg.cholesky(K)

        # Check for numerical issues
        #numerical_issues = jnp.isnan(L).any() or jnp.isinf(L).any()
        #if numerical_issues:
        #    return (-jnp.inf, jnp.zeros_like(params)) if self.kernel.eval_gradient else -jnp.inf

        # Compute the log-marginal likelihood using the alpha vector, the Cholesky decomposition (L), and the noise term
        alpha = cho_solve((L, True), self.y_train)
        log_marginal_likelihood = -0.5 * jnp.dot(self.y_train.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - len(self.X_train) * 0.5 * jnp.log(2 * jnp.pi)

        K_gradient = self.kernel.kernel_gradient(self.X_train, self.X_train, params)
        V = cho_solve((L, True), jnp.eye(K.shape[0]))[:, :, jnp.newaxis]
        log_marginal_likelihood_gradient = 0.5 * jnp.einsum("ijk,ijk->k", jnp.einsum("ij,ik->ijk", alpha, alpha) - V, K_gradient)

        return -log_marginal_likelihood, -log_marginal_likelihood_gradient
    '''
    def _loss_and_grads(self, params):
        #params = jnp.exp(params)

        #Compute the covariance matrix (K) using the kernel function and add the noise term
        K = self.kernel.kernel_function(self.X_train, self.X_train, params) + self.sigma_n**2 * jnp.eye(self.X_train.shape[0])

        # Compute the Cholesky decomposition (L) of the covariance matrix (K). 
        # If the decomposition fails, return -inf for the log-marginal likelihood and a zero gradient.
        try:
            L = jnp.linalg.cholesky(K)
        except jnp.linalg.LinAlgError:
            return (-jnp.inf, jnp.zeros_like(params)) if self.kernel.eval_gradient else -jnp.inf

        #Compute the log-marginal likelihood using the alpha vector, the Cholesky decomposition (L) and the noise term
        alpha = cho_solve((L, True), self.y_train)
        log_marginal_likelihood = -0.5 * jnp.dot(self.y_train, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - len(self.X_train) * 0.5 * jnp.log(2 * jnp.pi)

        K_gradient = self.kernel.kernel_gradient(self.X_train, self.X_train,params) #*params
        V = cho_solve((L, True), jnp.eye(K.shape[0])) [:, :, jnp.newaxis]
        log_marginal_likelihood_gradient = 0.5 * jnp.einsum("ijk,ijk->k", jnp.einsum("ij,ik->ijk", alpha, alpha) - V, K_gradient)

        return -log_marginal_likelihood, -log_marginal_likelihood_gradient
    '''
    
    
    # Return the negative log-marginal likelihood and its gradients
    def nll_value_and_grad(self,params):
        lml = self._log_marginal_likelihood(params)
        if self.kernel.eval_gradient:
            lml_grad = self._lml_grad(params)
            return -lml, -lml_grad
        else:
            return -lml, None


    #@partial(jit, static_argnums=(0,))
    #Not Used. Self-Implemented Adam Optimizer
    def _optimize_kernel_params_adam(self, max_iters=1000, step_size=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        m = jnp.zeros_like(self.kernel.param_values)
        v = jnp.zeros_like(self.kernel.param_values)
        t = 0
        best_loss = jnp.inf
        best_theta = self.kernel.param_values

        for _ in range(max_iters):
            t += 1
            loss, grads = self._loss_and_grads(self.kernel.param_values)
            best_theta = jnp.where(loss < best_loss, self.kernel.param_values, best_theta)
            best_loss = jnp.where(loss < best_loss, loss, best_loss)

            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.kernel.param_values = self.kernel.param_values - step_size * m_hat / (jnp.sqrt(v_hat) + eps)

            # Apply bounds
            self.kernel.param_values = jnp.array([jnp.clip(param, lower, upper) for (param, (lower, upper)) in zip(self.kernel.param_values, self.kernel.param_bounds)])

    # Optimize the kernel parameters using the specified optimizer
    def _optimize_kernel_params(self):
        def obj_func(params,):
            if self.kernel.eval_gradient:
                loss_val, grads = self._loss_and_grads(params)
                #loss_val, grads = self.nll_value_and_grad(params)
                return loss_val, grads
            else:
                loss_val, _ = self._loss_and_grads(params)
                #loss_val, _ = self.nll_value_and_grad(params)
                return loss_val

        initial_theta = self.kernel.param_values
        bounds = self.kernel.param_bounds

        theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds)
        self.kernel.param_values = theta_opt
        self.LML_ = -func_min

    # Make predictions with the Gaussian Process model
    @partial(jit, static_argnums=(0,2))
    def predict(self, X_star, return_cov = False):
        K_star = self.kernel(self.X_train, X_star)
        y_mean = jnp.dot(K_star.T, self.alpha)

        if return_cov:  
            K_star_star = self.kernel(X_star, X_star)
            v = cho_solve((cho_factor(self.K + self.sigma_n ** 2 * jnp.eye(len(self.y_train)))[0], True), K_star)
            y_cov = K_star_star - jnp.dot(K_star.T, v)
            return y_mean, y_cov
        else:
            return y_mean 
    
    def _log_prior(self, theta):
        log_prior = 0
        for param, (lower, upper) in zip(theta, self.kernel.param_bounds):
            log_prior += norm.logpdf(param, loc=(upper + lower) / 2, scale=(upper - lower) / 6)
        return log_prior

    def _log_posterior(self, theta):
        log_likelihood = self._log_marginal_likelihood(theta)
        log_prior = self._log_prior(theta)
        return log_likelihood + log_prior

    @staticmethod
    def metropolis_hastings_step(rng_key, theta_current, log_posterior_fn):
        log_posterior = jit(log_posterior_fn)  # Apply jit to the log_posterior function here
        rng_key, subkey = random.split(rng_key)
        proposal = random.normal(subkey, theta_current.shape) * 0.05 + theta_current
        log_prob_current = log_posterior(theta_current)
        log_prob_proposal = log_posterior(proposal)
        accept_ratio = jnp.exp(log_prob_proposal - log_prob_current)
        rng_key, subkey = random.split(rng_key)
        accept = random.bernoulli(subkey, jnp.minimum(1.0, accept_ratio))
        #print(theta_current,"theta_current")
        #print(proposal,"proposal")
        theta_next = jnp.where(accept, proposal, theta_current)
        #print(theta_next,"theta_next")
        return rng_key, theta_next.reshape(theta_current.shape)

    def mcmc_inference(self, initial_theta, n_samples=1000, burn_in=100, thinning=2):
        # Remove the jit here
        log_posterior = self._log_posterior

        # Initialize the chain
        rng_key = random.PRNGKey(0)
        theta_chain = [initial_theta]

        for _ in range(n_samples * thinning + burn_in):
            rng_key, theta_next = self.metropolis_hastings_step(rng_key, theta_chain[-1], log_posterior)
            theta_chain.append(theta_next)

        # Remove burn-in samples and apply thinning
        theta_chain = theta_chain[burn_in::thinning]

        self.kernel.param_values = jnp.mean(jnp.array(theta_chain), axis=0)

        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * jnp.eye(len(self.X_train))
        L = jnp.linalg.cholesky(self.K)
        self.alpha = cho_solve((L, True), self.y_train)

        return jnp.array(theta_chain)


# %%
class GaussianProcessClassifier(GaussianProcess):
    def __init__(self, *args, **kwargs):
        super(GaussianProcessClassifier, self).__init__(*args, **kwargs)

    def fit(self, X, y):
        # Convert the labels to {-1, 1}
        y = 2 * y - 1
        super(GaussianProcessClassifier, self).fit(X, y)

    def predict_proba(self, X_star, return_cov=False):
        y_mean, y_cov = super(GaussianProcessClassifier, self).predict(X_star, return_cov=True)
        proba = logistic(y_mean)
        if return_cov:
            return proba, y_cov
        else:
            return proba

    def predict(self, X_star):
        proba = self.predict_proba(X_star)
        labels = (proba > 0.5).astype(int)
        return labels

