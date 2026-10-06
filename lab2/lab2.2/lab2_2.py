import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from tests2_2 import get_system
except ImportError:
    from tests_2_2 import get_system


# Настройки границ и начальных точек для устойчивого вычисления ОДЗ
VARIANT_CONFIGS = {
    1: {"x_bounds": (-2.0, 4.0), "y_bounds": (-1.0, 3.0), "x0": 0.5, "y0": 0.5},
    2: {"x_bounds": (-2.0, 5.0), "y_bounds": (-1.0, 4.0), "x0": 0.5, "y0": 0.5},
    3: {"x_bounds": (-3.0, 7.0), "y_bounds": (-1.0, 5.0), "x0": 0.5, "y0": 0.5},
    4: {"x_bounds": (-0.9, 3.0), "y_bounds": (0.0, 3.0), "x0": 1.5, "y0": 1.2},
    5: {"x_bounds": (-0.9, 3.0), "y_bounds": (1.0, 4.0), "x0": 1.5, "y0": 2.2},
    6: {"x_bounds": (-0.9, 3.0), "y_bounds": (2.0, 5.0), "x0": 1.5, "y0": 3.2},
    7: {"x_bounds": (-2.5, 2.5), "y_bounds": (-2.5, 2.5), "x0": -1.5, "y0": 0.5},
    8: {"x_bounds": (-3.5, 3.5), "y_bounds": (-3.5, 3.5), "x0": -2.5, "y0": 0.5},
    9: {"x_bounds": (-4.5, 4.5), "y_bounds": (-4.5, 4.5), "x0": -3.5, "y0": 0.5},
    10: {"x_bounds": (-1.0, 3.0), "y_bounds": (-1.0, 3.0), "x0": 1.5, "y0": 1.8},
    11: {"x_bounds": (0.0, 4.0), "y_bounds": (0.0, 4.0), "x0": 2.5, "y0": 2.5},
    12: {"x_bounds": (1.0, 5.0), "y_bounds": (1.0, 5.0), "x0": 3.5, "y0": 3.5},
    13: {"x_bounds": (-2.2, 2.2), "y_bounds": (-1.2, 1.2), "x0": 0.5, "y0": 0.8},
    14: {"x_bounds": (-3.2, 3.2), "y_bounds": (-1.7, 1.7), "x0": 0.8, "y0": 0.8},
    15: {"x_bounds": (-4.2, 4.2), "y_bounds": (-2.2, 2.2), "x0": 1.0, "y0": 0.8},
    16: {"x_bounds": (-1.0, 1.5), "y_bounds": (-0.5, 1.5), "x0": 0.4, "y0": 0.7},
    17: {"x_bounds": (-1.0, 1.5), "y_bounds": (-0.5, 1.5), "x0": 0.3, "y0": 0.4},
    18: {"x_bounds": (-1.0, 1.5), "y_bounds": (-0.5, 1.5), "x0": 0.25, "y0": 0.3},
    19: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.01, 4.0), "x0": 1.5, "y0": 1.5},
    20: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.01, 4.0), "x0": 1.5, "y0": 1.2},
    21: {"x_bounds": (-3.0, 3.0), "y_bounds": (0.01, 4.0), "x0": 1.5, "y0": 0.9},
    22: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.8, "y0": 1.0},
    23: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.6, "y0": 0.7},
    24: {"x_bounds": (-1.3, 1.3), "y_bounds": (-2.0, 2.0), "x0": 0.5, "y0": 0.5},
    25: {"x_bounds": (-2.0, 3.0), "y_bounds": (-0.99, 3.0), "x0": 1.0, "y0": 1.0},
    26: {"x_bounds": (-2.0, 3.0), "y_bounds": (-1.99, 3.0), "x0": 1.0, "y0": 1.0},
    27: {"x_bounds": (-2.0, 3.0), "y_bounds": (-2.99, 3.0), "x0": 1.0, "y0": 1.0},
    28: {"x_bounds": (-1.0, 4.5), "y_bounds": (-0.5, 3.0), "x0": 2.0, "y0": 0.75},
    29: {"x_bounds": (-1.0, 5.5), "y_bounds": (-0.5, 3.0), "x0": 2.5, "y0": 1.0},
    30: {"x_bounds": (-1.0, 6.5), "y_bounds": (-0.5, 3.0), "x0": 3.0, "y0": 1.3},
}


def eval_safe(func, x, y):
    try:
        with np.errstate(all='ignore'):
            res = float(func(x, y))
            if np.isnan(res) or np.isinf(res):
                return None
            return res
    except Exception:
        return None


def newton_system(f1, f2, J_func, x0, y0, eps=1e-6, max_iter=100):
    x_curr, y_curr = x0, y0
    errors, path = [], [(x_curr, y_curr)]

    for k in range(1, max_iter + 1):
        v1, v2 = eval_safe(f1, x_curr, y_curr), eval_safe(f2, x_curr, y_curr)
        if v1 is None or v2 is None:
            raise ValueError(f"Точка ({x_curr:.3f}, {y_curr:.3f}) вне ОДЗ.")

        F = np.array([v1, v2], dtype=float)
        J = np.array(J_func(x_curr, y_curr), dtype=float)

        if np.isnan(J).any() or abs(np.linalg.det(J)) < 1e-12:
            raise ValueError("Вырожденный или некорректный Якобиан.")

        delta = np.linalg.solve(J, -F)

        tau = 1.0
        found = False
        while tau > 1e-5:
            x_next = x_curr + tau * delta[0]
            y_next = y_curr + tau * delta[1]
            if eval_safe(f1, x_next, y_next) is not None and eval_safe(f2, x_next, y_next) is not None:
                found = True
                break
            tau *= 0.5

        if not found:
            raise ValueError("Не удалось сделать шаг Ньютона в пределах ОДЗ.")

        err = np.linalg.norm([x_next - x_curr, y_next - y_curr])
        errors.append(err)
        path.append((x_next, y_next))

        F_next = np.array([f1(x_next, y_next), f2(x_next, y_next)], dtype=float)
        if err < eps or np.linalg.norm(F_next) < eps:
            return x_next, y_next, k, errors, path

        x_curr, y_curr = x_next, y_next

    raise RuntimeError("Метод Ньютона не сошелся.")


def simple_iteration_system(f1, f2, J_func, x0, y0, eps=1e-6, max_iter=500):
    J0 = np.array(J_func(x0, y0), dtype=float)
    if np.isnan(J0).any() or abs(np.linalg.det(J0)) < 1e-12:
        raise ValueError("Якобиан J(x0, y0) вырожден или вне ОДЗ.")

    Lambda = np.linalg.inv(J0)
    x_curr, y_curr = x0, y0
    errors, path = [], [(x_curr, y_curr)]

    for k in range(1, max_iter + 1):
        v1, v2 = eval_safe(f1, x_curr, y_curr), eval_safe(f2, x_curr, y_curr)
        if v1 is None or v2 is None:
            raise ValueError(f"Точка ({x_curr:.3f}, {y_curr:.3f}) вне ОДЗ.")

        F = np.array([v1, v2], dtype=float)
        delta = -Lambda @ F

        tau = 1.0
        found = False
        while tau > 1e-4:
            x_next = x_curr + tau * delta[0]
            y_next = y_curr + tau * delta[1]
            if eval_safe(f1, x_next, y_next) is not None and eval_safe(f2, x_next, y_next) is not None:
                found = True
                break
            tau *= 0.5

        if not found:
            raise ValueError(f"Расхождение простой итерации на шаге {k}.")

        err = np.linalg.norm([x_next - x_curr, y_next - y_curr])
        errors.append(err)
        path.append((x_next, y_next))

        F_next = np.array([f1(x_next, y_next), f2(x_next, y_next)], dtype=float)
        if err < eps or np.linalg.norm(F_next) < eps:
            return x_next, y_next, k, errors, path

        x_curr, y_curr = x_next, y_next

    raise RuntimeError("Метод простой итерации не сошелся.")


def main():
    try:
        variant = int(input("Введите номер варианта (1-30): "))
        x_sym, y_sym, sys_eqs = get_system(variant)
    except Exception as e:
        print(f"Ошибка выбора варианта: {e}")
        return

    f1_eq, f2_eq = sys_eqs[0], sys_eqs[1]
    J_matrix = sp.Matrix([f1_eq, f2_eq]).jacobian([x_sym, y_sym])

    f1 = sp.lambdify((x_sym, y_sym), f1_eq, 'numpy')
    f2 = sp.lambdify((x_sym, y_sym), f2_eq, 'numpy')
    J_func = sp.lambdify((x_sym, y_sym), J_matrix, 'numpy')

    cfg = VARIANT_CONFIGS.get(variant, {"x_bounds": (-3.0, 3.0), "y_bounds": (-3.0, 3.0), "x0": 1.0, "y0": 1.0})
    x0, y0 = cfg["x0"], cfg["y0"]
    eps = 1e-6

    print(f"\nВАРИАНТ {variant}")
    print(f"Система:\n 1) {f1_eq} = 0\n 2) {f2_eq} = 0")
    print(f"Начальное приближение: (x1_0, x2_0) = ({x0}, {y0}), eps = {eps}\n")

    # Решение
    try:
        xn, yn, kn, errs_n, path_n = newton_system(f1, f2, J_func, x0, y0, eps)
        f1_v, f2_v = f1(xn, yn), f2(xn, yn)
        print(f"[Метод Ньютона]\n  Решение: x1* = {xn:.6f}, x2* = {yn:.6f}\n  Итераций: {kn}\n  ||F|| = {np.linalg.norm([f1_v, f2_v]):.2e}\n")
    except Exception as e:
        print(f"[Метод Ньютона] Ошибка: {e}\n")
        xn, yn, kn, errs_n, path_n = None, None, 0, [], []

    try:
        xsi, ysi, ksi, errs_si, path_si = simple_iteration_system(f1, f2, J_func, x0, y0, eps)
        f1_si, f2_si = f1(xsi, ysi), f2(xsi, ysi)
        print(f"[Простая итерация]\n  Решение: x1* = {xsi:.6f}, x2* = {ysi:.6f}\n  Итераций: {ksi}\n  ||F|| = {np.linalg.norm([f1_si, f2_si]):.2e}\n")
    except Exception as e:
        print(f"[Простая итерация] Ошибка: {e}\n")
        xsi, ysi, ksi, errs_si, path_si = None, None, 0, [], []

    # График
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    xmin, xmax = cfg["x_bounds"]
    ymin, ymax = cfg["y_bounds"]
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 400), np.linspace(ymin, ymax, 400))

    with np.errstate(all='ignore'):
        Z1 = np.array(f1(X, Y), dtype=float)
        Z2 = np.array(f2(X, Y), dtype=float)

    Z1[~np.isfinite(Z1)] = np.nan
    Z2[~np.isfinite(Z2)] = np.nan

    legend_elems = []
    if np.nanmin(Z1) <= 0 <= np.nanmax(Z1):
        ax1.contour(X, Y, Z1, levels=[0], colors='red', linewidths=1.5)
        legend_elems.append(Line2D([0], [0], color='red', lw=1.5, label='f1 = 0'))
    if np.nanmin(Z2) <= 0 <= np.nanmax(Z2):
        ax1.contour(X, Y, Z2, levels=[0], colors='blue', linewidths=1.5)
        legend_elems.append(Line2D([0], [0], color='blue', lw=1.5, label='f2 = 0'))

    if path_n:
        path_arr = np.array(path_n)
        ax1.plot(path_arr[:, 0], path_arr[:, 1], 'g--s', label='Траектория Ньютона', markersize=4)
    if xn is not None and yn is not None:
        ax1.plot(xn, yn, 'ro', markersize=8, label=f'Корень ({xn:.3f}, {yn:.3f})')

    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title(f"Вариант {variant}: Система уравнений")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(loc='upper right')

    if errs_si:
        ax2.plot(range(1, ksi + 1), errs_si, 'r-o', label=f'Простая итерация ({ksi} итер.)')
    if errs_n:
        ax2.plot(range(1, kn + 1), errs_n, 'g-s', label=f'Ньютон ({kn} итер.)')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.set_title("Норма погрешности ||x_k - x_{k-1}||")
    ax2.set_xlabel("Номер итерации (k)")
    ax2.set_ylabel("Погрешность")
    ax2.legend()

    plt.tight_layout()
    filename = f"lab2_2_variant_{variant}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"График сохранен в файл: {filename}")


if __name__ == "__main__":
    main()