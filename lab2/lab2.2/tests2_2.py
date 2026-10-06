import sympy as sp

def get_system(variant: int):
    """
        x -> x1
        y -> x2
    """
    x, y = sp.symbols('x y')

    variants_data = {
        # Группа 1 
        1: (2, [(x**2 + 2**2)*y - 2**3, (x - 2/2)**2 + (y - 2/2)**2 - 2**2]),
        2: (3, [(x**2 + 3**2)*y - 3**3, (x - 3/2)**2 + (y - 3/2)**2 - 3**2]),
        3: (4, [(x**2 + 4**2)*y - 4**3, (x - 4/2)**2 + (y - 4/2)**2 - 4**2]),

        # Группа 2
        4: (1, [x - sp.cos(y) - 1, y - sp.log(x + 1, 10) - 1]),
        5: (2, [x - sp.cos(y) - 1, y - sp.log(x + 1, 10) - 2]),
        6: (3, [x - sp.cos(y) - 1, y - sp.log(x + 1, 10) - 3]),

        # Группа 3
        7: (2, [x**2 + y**2 - 2**2, x - sp.exp(y) + 2]),
        8: (3, [x**2 + y**2 - 3**2, x - sp.exp(y) + 3]),
        9: (4, [x**2 + y**2 - 4**2, x - sp.exp(y) + 4]),

        # Группа 4
        10: (1, [x - sp.cos(y) - 1, y - sp.sin(x) - 1]),
        11: (2, [x - sp.cos(y) - 2, y - sp.sin(x) - 2]),
        12: (3, [x - sp.cos(y) - 3, y - sp.sin(x) - 3]),

        # Группа 5
        13: (2, [x**2 / (2**2) + y**2 / (1**2) - 1, 2 * y - sp.exp(x) - x]),
        14: (3, [x**2 / (3**2) + y**2 / (1.5**2) - 1, 3 * y - sp.exp(x) - x]),
        15: (4, [x**2 / (4**2) + y**2 / (2**2) - 1, 4 * y - sp.exp(x) - x]),

        # Группа 6
        16: (2, [2 * x - sp.cos(y), 2 * y - sp.exp(x)]),
        17: (3, [3 * x - sp.cos(y), 3 * y - sp.exp(x)]),
        18: (4, [4 * x - sp.cos(y), 4 * y - sp.exp(x)]),

        # Группа 7
        19: (1, [x**2 - 2 * sp.log(y, 10) - 1, x**2 - 1 * x * y + 1]),
        20: (2, [x**2 - 2 * sp.log(y, 10) - 1, x**2 - 2 * x * y + 2]),
        21: (3, [x**2 - 2 * sp.log(y, 10) - 1, x**2 - 3 * x * y + 3]),

        # Группа 8
        22: (1, [1 * x**2 - x + y**2 - 1, y - sp.tan(x)]),
        23: (2, [2 * x**2 - x + y**2 - 1, y - sp.tan(x)]),
        24: (3, [3 * x**2 - x + y**2 - 1, y - sp.tan(x)]),

        # Группа 9
        25: (1, [1 * x**2 - y + y**2 - 1, x - sp.sqrt(y + 1) + 1]),
        26: (2, [2 * x**2 - y + y**2 - 2, x - sp.sqrt(y + 2) + 1]),
        27: (3, [3 * x**2 - y + y**2 - 3, x - sp.sqrt(y + 3) + 1]),

        # Группа 10
        28: (4, [sp.exp(x * y) + x - 4, x**2 - 4 * y - 1]),
        29: (5, [sp.exp(x * y) + x - 5, x**2 - 5 * y - 1]),
        30: (6, [sp.exp(x * y) + x - 6, x**2 - 6 * y - 1]),
    }

    if variant not in variants_data:
        raise ValueError(f"Вариант {variant} не найден в таблице.")

    a_val, sys_eqs = variants_data[variant]
    return x, y, sys_eqs