#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Diederik M. Roijers (Vrije Universiteit Brussel)
"""

import operator
import numpy as np
from copy import deepcopy


def pareto_front(vector_list):
    """
    Given a list of vectors, return the Pareto Front.
    I.e., remove all dominated vectors.
    :param vector_list:     a list of vectors
    :return:                a list of non-dominated vectors
    """
    result = []
    while len(vector_list) > 0:
        vset_prime = []
        x, *xs = vector_list
        vector_list = xs
        while len(vector_list) > 0:
            y, *ys = vector_list
            if all(map(operator.ge, x, y)):
                vector_list = ys
            else:
                if all(map(operator.ge, y, x)):
                    x = y
                    vector_list = ys
                else:
                    vset_prime.append(y)
                    vector_list = ys
        result.append(x)
        vector_list = list(filter(lambda z: (not all(map(operator.ge, x, z))),
                                  vset_prime))
    return result


def find_vector(vec, vlist):
    for i in range(len(vlist)):
        if np.array_equal(vlist[i], vec):
            return i
    return -1


def flattened_polynomial_term(vector, varPow, scalar):
    result = 1.0
    for i in range(len(varPow)):
        (v,exponent) = varPow[i]
        if vector[v] < 0 :
            return 0.0
        else:
            result = result*pow(vector[v], exponent)
    result = result * scalar
    return result


def flattened_polynomial_u(vector, terms, coefficients):
    result = 0.0
    for i in range(len(terms)):
        result = result + flattened_polynomial_term(vector, terms[i], coefficients[i])
    return result


def lambda_polynomial(terms, coefficients):
    return lambda vector: flattened_polynomial_u(vector, terms, coefficients)


def n_th_order_terms(n, n_vars):
    n_min1 = all_1th_tuples_list(n_vars)
    result = []
    result.extend(n_min1)
    for x in range(n-1):
        current = []
        for i in range(len(n_min1)):
            parent_term = n_min1[i]
            for j in range(n_vars):
                term = deepcopy(parent_term)
                add_1order_to_var_x(term,j)
                if term not in current:
                    current.append(term)
        result.extend(current)
        n_min1=current
    return result
            
    
def all_1th_tuples_list(n_vars):
    result = []
    for i in range(n_vars):
        result.append([(i,1)])
    return result

def add_1order_to_var_x(term, x):
    found = False
    for i in range(len(term)):
        if term[i][0] is x:
            term[i] = (x,term[i][1]+1)
            found = True
    if not found:
        term.append((x,1))
    term.sort(key=lambda tup: tup[0]) 


def random_polynomial_of_order_n(n, n_vars, min_c, max_c,seed=None):
    terms = n_th_order_terms(n, n_vars)
    random_state = np.random.RandomState(seed)
    coeffs = random_state.rand(len(terms))
    print(coeffs)
    coeffs = list(map(lambda x: x*(max_c-min_c)+min_c, coeffs))
    return lambda_polynomial(terms, coeffs)
    

if __name__ == '__main__':
    
    terms = [[(0,2)],[(0,1),(1,1)],[(1,2)]]
    coeffs= [1.0, 3.0, 0.5]
    functietje = lambda_polynomial(terms, coeffs)

    lst = []
    lst.append(np.array([1, 1]))
    lst.append(np.array([4, 1]))
    lst.append(np.array([5, -3]))
    lst.append(np.array([2, 1]))
    lst.append(np.array([1, 2]))
    lst.append(np.array([2, 2]))
    lst.append(np.array([2, 2]))
    lst.append(np.array([3, 1]))
    print(pareto_front(lst))
    
    for i in range(len(lst)):
        print(str(i)+": "+str(functietje(lst[i])))
        
    terms = all_1th_tuples_list(4)
    print(terms) 
    add_1order_to_var_x(terms[1], 0)
    print(terms) 
    
    print(n_th_order_terms(3,3))
    print(random_polynomial_of_order_n(3,2,0.5,1.2))
    

def hotel_utility(a, c):
    return lambda v: a * v[0] + c * np.log2((np.clip(v[1]*4+1, 1+1e-4, 5) - 1) / 2)

def bugs_utility():
    return lambda v: np.min(v, axis=-1)
