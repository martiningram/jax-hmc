import jax.numpy as jnp
from jax.ops import index_update
from jax.random import PRNGKey, normal, split, uniform
from jax.lax import cond, fori_loop
from jax import jit
from functools import partial
from jax.scipy.stats import norm


@jit
def draw_phi(key, M):
    # M is assumed diagonal

    key, subkey = split(key)

    draw = normal(subkey, (M.shape[0],))
    phi = jnp.sqrt(M) * draw

    return key, phi


@partial(jit, static_argnums=4)
def leapfrog(theta, phi, M, eps, grad_fun):

    cur_grad = grad_fun(theta)

    phi_star = phi + 0.5 * eps * cur_grad
    theta = theta + eps * (phi / M)
    cur_grad = grad_fun(theta)
    phi_star = phi_star + 0.5 * eps * cur_grad

    return theta, phi_star


@partial(jit, static_argnums=6)
def metropolis_hastings(theta, theta_star, phi, phi_star, M, key, log_posterior):

    log_prob_phi_star = jnp.sum(
        norm.logpdf(phi_star, jnp.zeros_like(theta), jnp.sqrt(M))
    )
    log_prob_phi = jnp.sum(norm.logpdf(phi, jnp.zeros_like(theta), jnp.sqrt(M)))

    log_prob_theta = log_posterior(theta)
    log_prob_theta_star = log_posterior(theta_star)

    log_numerator = log_prob_theta_star + log_prob_phi_star
    log_denominator = log_prob_theta + log_prob_phi

    log_accept_prob = log_numerator - log_denominator

    accept_prob = jnp.exp(log_accept_prob)

    key, subkey = split(key)

    rand_draw = uniform(subkey)

    new_theta = cond(
        rand_draw < accept_prob, lambda _: theta_star, lambda _: theta, theta
    )

    return key, new_theta, accept_prob


@partial(jit, static_argnums=2)
def body_fun_leapfrog(i, val, grad_fun):

    theta_star, phi_star = leapfrog(
        val["theta"], val["phi"], val["M"], val["eps"], grad_fun
    )

    val["theta"] = theta_star
    val["phi"] = phi_star

    return val


@partial(jit, static_argnums=5)
def iterate_leapfrogs(theta, phi, eps, M, L, grad_fun):

    init_val = {"theta": theta, "phi": phi, "eps": eps, "M": M}

    to_iterate = lambda i, val: body_fun_leapfrog(i, val, grad_fun)

    final_res = fori_loop(0, L, to_iterate, init_val)

    return final_res["theta"], final_res["phi"]


@partial(jit, static_argnums=(2, 3))
def body_fun_sampling(i, val, grad_fun, log_posterior):

    key, phi = draw_phi(val["key"], val["M"])

    theta_star, phi_star = iterate_leapfrogs(
        val["theta"], phi, val["eps"], val["M"], val["L"], grad_fun
    )

    key, theta, accept_prob = metropolis_hastings(
        val["theta"], theta_star, phi, phi_star, val["M"], key, log_posterior
    )

    val["samples"] = index_update(val["samples"], i, theta)
    val["theta"] = theta
    val["key"] = key

    return val


def draw_samples(theta_dim, eps, M, L, key, n_samples, grad_fun, log_posterior):

    theta = jnp.zeros(theta_dim)

    init_val = {
        "theta": theta,
        "eps": eps,
        "M": M,
        "key": key,
        "L": L,
        "samples": jnp.zeros((n_samples, theta_dim)),
    }

    to_iterate = lambda i, val: body_fun_sampling(i, val, grad_fun, log_posterior)

    final_res = fori_loop(0, n_samples, to_iterate, init_val)

    return final_res
