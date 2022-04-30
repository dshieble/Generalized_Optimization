from collections import namedtuple

import numpy as np
from numpy.polynomial import Polynomial
import sympy
from functools import reduce


Results = namedtuple("Results", "best_vals best_objective_value last_vals last_objective_value")


def size_one_steps(num_dict, **kwargs):
    """
    Given an integer-valued vector, return the integer-valued vector that has a maximum dimension absolute value of
        1 and points in as similar a direction as possible
    """
    return {key: int(np.sign(num)) for key, num in num_dict.items()}


def random_sos_polynomial(
    max_vars,
    max_terms,
    max_symbols_per_term,
    max_coeff,
    max_offset
):
    """
    Generate a random sympy polynomial that can be expressed as a sum of squares
    Args:
        max_vars: The maximum number of unique variables in the polynomial
        max_terms: The maximum number of terms in the polynomial
        max_symbols_per_term: The maximum number of variables in each term
        max_coeff: The maximum coefficient on any term
        max_offset: The maximum value of the offset (constant) term
        max_num_vars: 
    Returns:
        A polynomial
    """
    num_vars = np.random.randint(1, max_vars)
    num_terms = np.random.randint(1, max_terms)
    symbols = [sympy.symbols('x_{}'.format(i)) for i in range(num_vars)]
    poly = 0
    for term in range(num_terms):
        included_symbols = np.random.choice(symbols, size=1 + int(np.random.random() * max_symbols_per_term), replace=True)
        variable_term = reduce(lambda a,b: a * b, included_symbols)

        # Generate a coefficient
        coeff = int(np.random.random()*max_coeff) + 1

        # Generate an offset term between -max_offset, max_offset
        offset = np.random.randint(-max_offset, max_offset)

        # We square all of the terms since a sum-of-square polynomial is positive in Rn 
        #    https://en.wikipedia.org/wiki/Positive_polynomial#Examples
        poly += (coeff * variable_term + offset)**2
    return poly.as_poly(), poly.as_poly().gens


def sample_gen_vals(poly, max_scale=5):
    """
    Given a sympy polynomial, choose scale uniformly from [1, max_scale], then generate a random value for each variable
        by generating an exponent in (-scale, scale) and selecting an integer in (-2^scale, 2^scale)
    """
    scale = np.random.randint(low=1, high=max_scale)
    sampled_nums = np.random.randint(
        low=0, high=scale, size=np.array([len(poly.gens)]))
    nums = [np.random.choice([-1, 1]) * (2**s) for s in sampled_nums]
    return {g: v for g, v in zip(poly.gens, nums)}


def random_search(poly, steps):
    """
    Use sample_gen_vals to randomly generate values and test them in the polynomial
    Args:
        poly: The sympy polynomial to optimize
        steps: Steps to take
    Returns:
        optimization results
    """
    best_vals, best_objective_value = None, np.inf
    for i in range(steps):
        gen_vals = sample_gen_vals(poly=poly)
        objective_value = poly.eval(gen_vals)
        
        # Update if this loss is better than what we have seen before
        if objective_value < best_objective_value:
            best_vals, best_objective_value = gen_vals, objective_value
    return Results(
        best_vals=best_vals,
        best_objective_value=best_objective_value,
        last_vals=gen_vals,
        last_objective_value=objective_value)


def gradient_descent(
        poly,
        steps,
        reduce_fn=size_one_steps,
        min_value=-1e50,
        max_value=1e10,
        verbosity=None):
    """
    Run gradient descent on a sympy polynomial
    Args
        poly: The sympy polynomial to optimize
        steps: Steps to take
        reduce_fn: The function to modify the gradient
        min_value: The point to stop the optimization because we have converged
        max_value: The point to stop the optimization because we have diverged
        verbosity: If not None, the number of iterations at which to log the objective
    Returns:
        Optimization results
    """
    start = sample_gen_vals(poly)

    assert list(start.keys()) == list(poly.gens)

    gen_vals = {k: v for k, v in start.items()}
    best_vals, best_objective_value = gen_vals, np.inf
    for i in range(steps):
        objective_value = poly.eval(gen_vals)

        if verbosity is not None and i % (steps / verbosity) == 0:
            print(float(objective_value))

        if objective_value < min_value or objective_value >= 1e10:
            break
        
        raw_grad = {var: poly.diff(var).eval(gen_vals) for var in gen_vals.keys()}
        reduced_grad = reduce_fn(num_dict=raw_grad, poly=poly, gen_vals=gen_vals) if reduce_fn is not None else raw_grad
    
        gen_vals = {var: gen_vals[var] - reduced_grad[var] for var, val in gen_vals.items()}
        # Update if this loss is better than what we have seen before
        if objective_value < best_objective_value:
            best_vals, best_objective_value = gen_vals, objective_value
        
    return Results(
        best_vals=best_vals,
        best_objective_value=best_objective_value,
        last_vals=gen_vals,
        last_objective_value=objective_value)



  
    
