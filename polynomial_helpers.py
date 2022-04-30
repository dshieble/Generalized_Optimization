from collections import namedtuple
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import sympy
from scipy.spatial.distance import cosine
from tqdm import tqdm
from sympy.solvers.solveset import linsolve
from sympy.printing.latex import LatexPrinter, print_latex


DynamicalSystemResult = namedtuple("DynamicalSystemResult", "t_values, s_values, coefficient_dict")


def get_dynamical_system_result(x, polynomial_to_optimize, max_t):
    """
    Args:
        x: sympy variable
        polynomial_to_optimize: A sympy polynomial constructed on x
        max_t: The maximum time step
    Return:
        A DynamicalSystemResult describing the trajectory of the optimization process for optimizing
            polynomial_to_optimize
    """
    t_values = range(1, max_t + 1)
    
    # Get the equations to solve
    equations, coefficient_dict = get_equations_from_polynomial(
        x=x,
        polynomial_to_optimize=polynomial_to_optimize,
        max_t=max_t)
    
    # Solve the equations
    solution_dict = get_solution_dict(equations=equations,  coefficient_dict=coefficient_dict)
    
    # Extract the system trajectory polynomial from the solution
    coefficient_dict = {i: solution_dict[coefficient_dict[i]] for i in range(len(solution_dict))}
    
    # Evaluate the polynomial at all points
    s_values = [evaluate_s(t_value=t_value, coefficient_dict=coefficient_dict) for t_value in t_values]
    return DynamicalSystemResult(t_values=t_values, s_values=s_values, coefficient_dict=coefficient_dict)


def get_equations_from_polynomial(x, polynomial_to_optimize, max_t):
    """
    Given a polynomial to optimize, compute the set of equations that the solution trajectory would satisfy
    Args:
        x: sympy variable
        polynomial_to_optimize: A sympy polynomial constructed on x
        max_t: The maximum time step
    Return:
        A tuple of (equations to solve, dictionary mapping indices to sympy variables in equation)
    """
    P = lambda x: polynomial_to_optimize.as_poly().eval(x)
    dP = lambda x: polynomial_to_optimize.diff().as_poly().eval(x)    

    # The degree of the polynomial s
    num_symbols = 2*max_t + 1

    # The coefficients of the polynomial s
    coefficient_dict = {i: sympy.symbols("a{}".format(i)) for i in range(num_symbols)}

    # The equations to require that the derivative of s is dl/dx
    dldx_dsdt_equations = [get_dldx_dsdt_equation(t=i, coefficient_dict=coefficient_dict, dP=dP) for i in range(1, max_t + 1)]

    # The equations to require that s passes through gradient descent
    dynamical_system_equations = [
        get_dynamical_system_equation(t=i, coefficient_dict=coefficient_dict, dP=dP) for i in range(1, max_t + 1)]
    return dldx_dsdt_equations + dynamical_system_equations, coefficient_dict


def get_dynamical_system_equation(t, coefficient_dict, dP):
    """
    Returns the s(t+1) = s(t) + dl/dx(s(t)) equation that we implicitly set equal to 0. This is the equation that enforces that the integer polynomial s passes through the trajectory of integer gradient descent
        t: integer
        coefficient_dict: dictionary {index: sympy variable}
        dP: linear function
    
    This returns the following expression, which we interpret as being set equal to 0
        a0(1 - 1 - dP(1)) + a1((t+1) - t - dP(t)) + a2((t+1)^2 - t^2 - dP(t^2)) + ... + an((t+1)^n - t^n - dP(t^n))
    """
    out = 0
    for i in range(len(coefficient_dict)):
        out += coefficient_dict[i]*(((t + 1)**i) - (t**i) + dP(t**i))
    return out


def get_dldx_dsdt_equation(t, coefficient_dict, dP):
    """
    Returns the dl/dx(s(t)) = ds/dx(t) equation that we implicitly set equal to 0. This is the equation that enforces that at t the derivative of the integer polynomial ds/dx(t) is equal to the gradient of the loss dl/dx(s(t))
        t: integer
        coefficient_dict: dictionary {index: sympy variable}
        dP: linear function
    
    This returns the following expression, which we interpret as being set equal to 0
        (dP(a0) + a1) + (dP(a1) + 2*a2)*t + (dP(a2) + 3*a3)*(t**2) + (dP(a3) + 4*a4)*(t**3) + dP(a4)*(t**4)
    """
    out = 0
    for i in range(len(coefficient_dict) - 1):
        out += (dP(coefficient_dict[i]) + (i+1)*coefficient_dict[i+1])*(t**i)
    out += dP(coefficient_dict[len(coefficient_dict) - 1]*(t**(len(coefficient_dict) - 1)))
    return out


def evaluate_s(t_value, coefficient_dict):
    """
    Evaluate the sympy polynomial s 
        t_value: a single number or sympy variable
        coefficient_dict: a dictionary {index: number or sympy variable}
    Return:
        The output of evaluating the polynomial defined by coefficient_dict at t_value
    """
    out = 0
    for i in range(len(coefficient_dict)):
        m = 1
        for j in range(i):
            m = m*t_value
        out += coefficient_dict[i]*m
    return out


def choose_solution(underspecified_solution_values):
    # Replace all of the variable terms in a list of sympy expressions with 1
    symbol_dict = {
        x: sympy.Rational(np.random.randint(-10, 10))
        for x in underspecified_solution_values if isinstance(x, sympy.core.symbol.Symbol)}
    solution_values = []
    for x in list(underspecified_solution_values):
        if isinstance(x, sympy.core.numbers.Number):
            solution_values.append(x)
        elif isinstance(x, sympy.core.symbol.Symbol):
            solution_values.append(symbol_dict[x])
        else:
            solution_values.append(x.as_poly().eval(symbol_dict))
    return solution_values


def get_solution_dict(equations, coefficient_dict):
    """
    Solve the system of diophantine equations by solving the linear system and upscaling the solution
        equations: A system of equations
        coefficient_dict: a dictionary {index: number or sympy variable}
    Returns:
        A dictionary from sympy variable to value
    """
    
    # Solve as a linear equation. Results may be rational
    underspecified_solution_values = list(linsolve(equations, list(coefficient_dict.values())))[0]
    raw_rational_solution_values = choose_solution(underspecified_solution_values=underspecified_solution_values)
    raw_rational_solution_dict = {a_var: value for a_var, value in zip(coefficient_dict.values(), raw_rational_solution_values)} 
    
    # Convert rational solution to integer solution with least common multiple
    raw_denominator_lcm = np.lcm.reduce([s.q for s in raw_rational_solution_dict.values()])
    solution_values = [value*raw_denominator_lcm for value in raw_rational_solution_values]
    solution_dict = {a_var: value for a_var, value in zip(coefficient_dict.values(), solution_values)} 

    # Verify the solution
    for equation in equations:
        poly = equation.as_poly()
        
        # Verify solution generated by solver
        assert poly.eval({k: raw_rational_solution_dict[k] for k in poly.gens}) == 0, equation
        assert poly.eval({k: solution_dict[k] for k in poly.gens}) == 0, equation

    # Verify that the solution values are integers
    for s in solution_values:
        assert s.q == 1, solution_values
    
    return solution_dict

