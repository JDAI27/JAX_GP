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
#from jax.scipy.linalg import cho_factor, cho_solve
config.update('jax_enable_x64', True)


#%%
class GaussianProcess:
    def __init__(self, kernel=None, sigma_n=1e-5,is_mogp = False,output_dim =1, optimizer='adam', **optimizer_kwargs):
        self.kernel = kernel  # Kernel function for the Gaussian Process
        self.sigma_n = sigma_n  # Noise level (variance) for the observations
        self.optimizer = optimizer  # Optimizer for kernel parameter optimization
        self.optimizer_kwargs = optimizer_kwargs  # Additional arguments for the optimizer
        self.X_train = None  # Training data (features)
        self.y_train = None  # Training data (targets)
        self.alpha = None  # Alpha vector for making predictions
        self.K = None  # Covariance matrix of the training data
        self.LML_ = None # Log Marginal Likelihood of the Gaussian Process
        self.is_mogp = is_mogp # If the model is a multi-output Gaussian Process
        self.output_dim = output_dim  # Number of outputs (dimensions) of the Gaussian Process

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
        # Use Scipy's BFGS optimizer to minimize the objective function
        res = minimize(obj_func, initial_theta, method='BFGS')
        
        # Get the optimized kernel parameters
        theta_opt = res.x
        
        # Clip the optimized parameters to the specified bounds
        bounds = jnp.array(bounds)
        theta_opt = jnp.array([jnp.clip(param, bounds[i, 0], bounds[i, 1]) for i, param in enumerate(theta_opt)])
        
        # Compute the minimum value of the objective function
        func_min = obj_func(theta_opt)[0]
        
        # Return the optimized kernel parameters and the minimum value of the objective function
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
            
        self.X_train = jnp.atleast_2d(X) # Store the training data (inputs)
        if self.is_mogp:
            self.y_train = jnp.atleast_2d(y)
            self.y_vec = jnp.atleast_2d(y)[:,:-1]
        else:
            self.y_train = jnp.atleast_2d(y).T  # Store the training data (targets)
            self.y_vec = self.y_train

        # Optimize kernel parameters
        self._optimize_kernel_params()

        # Compute alpha with optimized kernel parameters
        self.K = self.kernel(self.X_train, self.X_train) + self.sigma_n**2 * jnp.eye(self.X_train.shape[0])
        # L = jnp.linalg.cholesky(self.K)
        L, lower = cho_factor(self.K)
        self.alpha = cho_solve((L, True), self.y_vec)

    # Compute the log-marginal likelihood given kernel parameters
    @partial(jit, static_argnums=(0,))
    def _log_marginal_likelihood(self, params):
        params = jnp.array(params)
        X, y = self.X_train, self.y_vec
        #print(params.shape)
        #print(self.kernel.kernel_function(X, X,params).shape)
        K = self.kernel.kernel_function(X, X,params)+ self.sigma_n**2 * jnp.eye(len(X))
        try:
            # L = jnp.linalg.cholesky(K)
            L, lower = cho_factor(K + self.sigma_n**2 * jnp.eye(self.X_train.shape[0]))
        except jnp.linalg.LinAlgError:
            return (-jnp.inf, jnp.zeros_like(self.kernel.param_values)) if self.kernel.eval_gradient else -jnp.inf
        alpha = cho_solve((L, True), y)
        log_likelihood = jnp.sum(-0.5 * jnp.dot(y.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - X.shape[0] / 2 * jnp.log(2 * jnp.pi))
        return log_likelihood

    # Compute the loss and gradients of the log-marginal likelihood given kernel parameters
    @partial(jit, static_argnums=(0,))
    def _loss_and_grads(self, params):
        # Compute the covariance matrix (K) using the kernel function and add the noise term
        K = self.kernel.kernel_function(self.X_train, self.X_train, params) + self.sigma_n**2 * jnp.eye(self.X_train.shape[0])

        # Compute the Cholesky decomposition (L) of the covariance matrix (K).
        #L = jnp.linalg.cholesky(K)
        
        L, lower = cho_factor(K + self.sigma_n**2 * jnp.eye(self.X_train.shape[0]))

        # Check for numerical issues
        #numerical_issues = jnp.isnan(L).any() or jnp.isinf(L).any()
        #if numerical_issues:
        #    return (-jnp.inf, jnp.zeros_like(params)) if self.kernel.eval_gradient else -jnp.inf

        # Compute the log-marginal likelihood using the alpha vector, the Cholesky decomposition (L), and the noise term
        alpha = cho_solve((L, True), self.y_vec)
        log_marginal_likelihood = -0.5 * jnp.dot(self.y_vec.T, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - len(self.X_train) * 0.5 * jnp.log(2 * jnp.pi)

        K_gradient = self.kernel.gradient(self.X_train, self.X_train, params)
        V = cho_solve((L, True), jnp.eye(K.shape[0]))[:, :, jnp.newaxis]
        log_marginal_likelihood_gradient = 0.5 * jnp.einsum("ijk,ijk->k", jnp.einsum("ij,ik->ijk", alpha, alpha) - V, K_gradient)

        return -log_marginal_likelihood, -log_marginal_likelihood_gradient


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

        if self.is_mogp:
            y_mean = jnp.vstack((y_mean, X_star[:, -1][:, None]))



        if return_cov:  
            K_star_star = self.kernel(X_star, X_star)
            v = cho_solve((cho_factor(self.K + self.sigma_n**2 * jnp.eye(self.X_train.shape[0])), True), K_star)
            y_cov = K_star_star - jnp.dot(K_star.T, v)
            return y_mean, y_cov
        else:
            return y_mean 
    
    @partial(jit, static_argnums=(0,))
    def _log_prior(self, theta):
        log_prior = 0
        for param, (lower, upper) in zip(theta, self.kernel.param_bounds):
            log_prior += norm.logpdf(param, loc=(upper + lower) / 2, scale=(upper - lower) / 6)
        return log_prior

    @partial(jit, static_argnums=(0,))
    def _log_posterior(self, theta):
        log_likelihood = self._log_marginal_likelihood(theta)
        log_prior = self._log_prior(theta)
        #print(log_likelihood,"log_likelihood")
        #print(log_prior,"log_prior")
        return log_likelihood + log_prior

    @staticmethod
    def metropolis_hastings_step(rng_key, theta_current, log_posterior):
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

    @staticmethod
    def adaptive_mcmc_step(rng_key, theta_current, log_posterior, cov_matrix, adaptation_rate=0.1):
        rng_key, subkey = random.split(rng_key)

        # Generate proposal using current covariance matrix
        proposal_noise = random.multivariate_normal(subkey, jnp.zeros(theta_current.shape), cov_matrix)
        proposal = theta_current + proposal_noise

        log_prob_current = log_posterior(theta_current)
        log_prob_proposal = log_posterior(proposal)
        accept_ratio = jnp.exp(log_prob_proposal - log_prob_current)

        rng_key, subkey = random.split(rng_key)
        accept = random.bernoulli(subkey, jnp.minimum(1.0, accept_ratio))

        theta_next = jnp.where(accept, proposal, theta_current)
        
        # Update the covariance matrix using the adaptation_rate
        delta_theta = jnp.outer(theta_next - theta_current, theta_next - theta_current)
        updated_cov_matrix = (1 - adaptation_rate) * cov_matrix + adaptation_rate * delta_theta

        return rng_key, theta_next.reshape(theta_current.shape), updated_cov_matrix

    @staticmethod
    def leapfrog_step(theta, momentum, log_posterior_grad, step_size):
        momentum = momentum - 0.5 * step_size * log_posterior_grad(theta)
        theta = theta + step_size * momentum
        momentum = momentum - 0.5 * step_size * log_posterior_grad(theta)
        return theta, momentum

    @staticmethod
    def hamiltonian_monte_carlo_step(rng_key, theta_current, log_posterior_fn, step_size=1000, num_leapfrog_steps=1000):
        log_posterior_grad = jit(grad(log_posterior_fn))  # Apply grad and jit to the log_posterior function here
        rng_key, subkey = random.split(rng_key)

        # Sample random momentum from a standard normal distribution
        momentum_current = random.normal(subkey, theta_current.shape)

        # Perform leapfrog integration
        theta_proposal, momentum_proposal = theta_current, momentum_current
        for _ in range(num_leapfrog_steps):
            theta_proposal, momentum_proposal = GaussianProcess.leapfrog_step(theta_proposal, momentum_proposal, log_posterior_grad, step_size)

        # Calculate Hamiltonian (Energy) for the current and proposal states
        hamiltonian_current = -log_posterior_fn(theta_current) + 0.5 * jnp.sum(momentum_current ** 2)
        hamiltonian_proposal = -log_posterior_fn(theta_proposal) + 0.5 * jnp.sum(momentum_proposal ** 2)

        # Calculate the acceptance probability
        accept_ratio = jnp.exp(hamiltonian_current - hamiltonian_proposal)
        rng_key, subkey = random.split(rng_key)
        accept = random.bernoulli(subkey, jnp.minimum(1.0, accept_ratio))

        # Update the state based on acceptance
        theta_next = jnp.where(accept, theta_proposal, theta_current)

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
        # L = jnp.linalg.cholesky(self.K)
        L, lower = cho_factor(self.K + self.sigma_n**2 * jnp.eye(self.X_train.shape[0]))
        self.alpha = cho_solve((L, True), self.y_vec)

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

