import sympy as sp

def get_equations():
    """ все 30 вариков выражений"""
    x = sp.Symbol('x')
    equations = {
        1: 2**2 - x**2 - 0.5,
        2: sp.log(x + 2) - x**2,
        3: sp.sqrt(1-x**2) - sp.exp(x) + 0.1,
        4: x**3 + x**2 - x - 0.5,
        5: sp.cos(x) + 0.25*x - 0.5,
        6: sp.exp(x) - 2*x - 2,
        7: 2**x + x**2 - 2,
        8: sp.log(x + 1) - 2*x**2 + 1,
        9: x**3 + x**2 - 2*x - 1,
        10: sp.sin(x) - 2*x**2 + 0.5,
        11: sp.exp(x) - x**3 + 3*x**2 - 2*x - 3,
        12: 3**x - 5*x**2 + 1,
        13: sp.log(x + 1) - 2*x + 0.5,
        14: x**3 - 2*x**2 - 10*x + 15,
        15: sp.sin(x) - x**2 + 1,
        16: x * sp.exp(x) + x**2 - 1,
        17: 4**x - 5*x - 2,
        18: sp.log(x + 1) - x**3 + 1,
        19: x**4 - 2*x - 1,
        20: sp.tan(x) - 5*x**2 + 1,
        21: 3 * sp.sqrt(x + 1) - sp.exp(x) - 0.5,
        22: 10**x - 5*x - 2,
        23: sp.log(x + 2) - x**4 + 0.5,
        24: x**6 - 5*x - 2,
        25: sp.sqrt(x + 2) - 2 * sp.cos(x),
        26: sp.log(x + 1, 10) - x + 0.5,
        27: x**6 - 5*x**3 - 2,
        28: sp.log(2*x + 1, 10) - x**3 + 1,
        29: x**5 - 7*x**2 + 3,
        30: x * sp.log(x + 2, 10) + x**2 - 1
    }
    return x, equations