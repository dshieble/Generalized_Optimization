import numpy as np
from numpy.polynomial import Polynomial
import sympy
from functools import reduce
from scipy.spatial.distance import cosine
from tqdm import tqdm
from collections import namedtuple

NumericType = namedtuple("NumericType", "int rational")



######################################## Integer Helpers ########################################
def size_one_steps(num_dict, **kwargs):
    """
    Given an integer-valued vector, return the integer-valued vector that has a maximum dimension magnitude of
        max_value and points in as similar a direction as possible
    """
    return {key: int(np.sign(num)) for key, num in num_dict.items()}



def reduce_nums(num_dict, **kwargs):
    try:
        """
        Given an integer-valued vector, return the smallest integer-valued vector that points in the same direction
        """
        div = None
        while div is None or div >= 2:
            div = np.gcd.reduce(np.array([int(k) for k in num_dict.values()], dtype=np.int32))
            assert div != 0
            num_dict = {key: int(num / div) for key, num in num_dict.items()}
    except TypeError as e:
        print(num_dict)
        raise e
    except OverflowError:
        return num_dict
    return num_dict


def reduce_nums_approx(num_dict, max_value=10, **kwargs):
    """
    Given an integer-valued vector, return the integer-valued vector that has a maximum dimension magnitude of
        max_value and points in as similar a direction as possible
    """
    div = max(num_dict.values()) / max_value
    if div == 0:
        return num_dict
    return {key: int(num / div) for key, num in num_dict.items()}


def reduce_nums_log(num_dict, max_value=100, **kwargs):
    """
    Given an integer-valued vector, return the integer-valued vector that has a maximum dimension magnitude of
        max_value and points in as similar a direction as possible
    """
    return {key: np.sign(num)*int(np.log(np.abs(1 + float(num)))) for key, num in num_dict.items()}



def line_search_reduce_nums_approx(num_dict, poly, gen_vals):
    """
    Use a line search to find optimal learning rate
        num_dict: Map from variable to gradient value
        poly: sympy polynomial to minimize
        gen_vals: Map from variable to current value
    """
    best_objective_fn, best_mv = np.inf, None
    for mv in [1, 5, 10, 100, 1000]:
        reduced_num_dict = reduce_nums_approx(num_dict=num_dict, max_value=mv) 
        objective_fn = poly.eval(
            {var: gen_vals[var] - reduced_num_dict[var] for var, val in gen_vals.items()})
        if objective_fn <= best_objective_fn:
            best_objective_fn = objective_fn
            best_mv = mv
    return reduce_nums_approx(num_dict=num_dict, max_value=best_mv) 


def line_search_learning_rate(num_dict, poly, gen_vals):
    """
    Use a line search to find optimal learning rate
        num_dict: Map from variable to gradient value
        poly: sympy polynomial to minimize
        gen_vals: Map from variable to current value
    """
    best_objective_fn, best_lr = np.inf, None
    for lr in [sympy.Rational(10), sympy.Rational(1), sympy.Rational(0.1), sympy.Rational(0.01), sympy.Rational(0.001)]:
        objective_fn = poly.eval({var: gen_vals[var] - lr*num_dict[var] for var, val in gen_vals.items()})
        if objective_fn <= best_objective_fn:
            best_objective_fn = objective_fn
            best_lr = lr
    return {k: best_lr * v for k, v in num_dict.items()}




######################################## Random helpers ########################################

def random_sos_polynomial(max_vars, max_terms, numeric_type, max_symbols_per_term, max_coeff, max_offset):
    """
    Generate a random sympy polynomial
    """
    num_vars = np.random.randint(1, 10)
    num_terms = np.random.randint(1, 10)
    symbols = [sympy.symbols('x_{}'.format(i)) for i in range(num_vars)]
    poly = 0
    for term in range(num_terms):
        included_symbols = np.random.choice(symbols, size=1 + int(np.random.random()*max_symbols_per_term), replace=True)
        variable_term = reduce(lambda a,b: a * b, included_symbols)

        # Generate a coefficient
        if numeric_type == NumericType.int:
            coeff = int(np.random.random()*max_coeff) + 1
        elif numeric_type == NumericType.rational:
            coeff = sympy.Rational(int(np.random.random()*max_coeff) + 1, int(np.random.random()*max_coeff) + 1)
        else:
            raise ValueError(numeric_type)
                
        # Generate an offset term
        offset = int(np.random.random()*2*max_offset - max_offset)
        
        # We square all of the terms since a sum-of-square polynomial is positive in Rn 
        #    https://en.wikipedia.org/wiki/Positive_polynomial#Examples
        poly += (coeff * variable_term + offset)**2
    return poly.as_poly(), poly.as_poly().gens


def sample_gen_vals(poly, scale, numeric_type):
    """
    Given a sympy polynomial, generate random variable values
    """
    sampled_nums = np.random.randint(
        low=0, high=scale, size=np.array([len(poly.gens)]))
    raw_nums = [(-1)**(np.random.randint(low=1,high=3)) * (2**s) for s in sampled_nums]
    if numeric_type == NumericType.int:
        nums = raw_nums
    elif numeric_type == NumericType.rational:
        denominators = np.random.randint(
            low=1, high=scale, size=np.array([len(poly.gens)]))
        nums = [sympy.Rational(n, d) for n, d in zip(raw_nums, denominators)]
    else:
        raise ValueError(numeric_type)
    return {g: v for g, v in zip(poly.gens, nums)}




def random_search(poly, numeric_type, steps=1000, scale=50):
    """
    Uniformly generate exponents in (-scale, scale), then generates
        random numbers from (-2^scale, 2^scale) and test them in the polynomial
    """
    best_vals, best_objective_value = None, np.inf
    for i in range(steps):
        gen_vals = sample_gen_vals(poly=poly, scale=scale, numeric_type=numeric_type)
        objective_value = poly.eval(gen_vals)
        
        # Update if this loss is better than what we have seen before
        if objective_value < best_objective_value:
            best_vals, best_objective_value = gen_vals, objective_value
    return best_vals, best_objective_value




######################################## Gradient Descent ########################################

def gradient_descent(poly, start, steps, reduce_fn=None, min_value=-1e10, verbosity=None):
    """
    Run gradient descent on a sympy polynomial
        poly: sympy polynomial
        start: starting point for gradient descent
        steps: steps to take
        reduce_fn: the function to modify the gradient (e.g. multiply by learning rate)
    Returns:
        gen_vals: The map from variable to value 
        best_objective_value
        final_objective_value: The final objective value
    """
    assert list(start.keys()) == list(poly.gens)

    gen_vals = {k: v for k, v in start.items()}
    best_objective_value = np.inf
    for i in range(steps):
        objective_value = poly.eval(gen_vals)
        best_objective_value = min(best_objective_value, objective_value)

        if verbosity is not None and i % (steps / verbosity) == 0:
            print(float(objective_value))

        if objective_value < min_value or objective_value >= 1e10:
            break
        
        raw_grad = {var: poly.diff(var).eval(gen_vals) for var in gen_vals.keys()}
        reduced_grad = reduce_fn(num_dict=raw_grad, poly=poly, gen_vals=gen_vals) if reduce_fn is not None else raw_grad
    
        gen_vals = {var: gen_vals[var] - reduced_grad[var] for var, val in gen_vals.items()}
    return gen_vals, best_objective_value, objective_value


def gd_with_restart(poly, restarts, steps, numeric_type, scale, reduce_fn=None, min_value=-1e10, verbosity=None):
    """
    Run gradient descent with random restarts on a sympy polynomial
        poly: sympy polynomial
        restarts: number of random restarts
        steps: total steps to take across all restarts
        numeric_type
        scale: search scale for random number generation
        reduce_fn: the function to modify the gradient (e.g. multiply by learning rate)
    Returns:
        gen_vals: The map from variable to value 
        best_objective_value
    """
    best_vals, best_objective_value = None, np.inf

    for i in range(restarts):
        start = sample_gen_vals(poly, scale, numeric_type)
        gen_vals, objective_value, _ = gradient_descent(
            poly=poly,
            start=start,
            steps=steps,
            min_value=min_value,
            verbosity=verbosity,
            reduce_fn=reduce_fn)

    # Update if this loss is better than what we have seen before
    if objective_value < best_objective_value:
        best_vals, best_objective_value = gen_vals, objective_value
    return best_vals, best_objective_value
        

  
    
