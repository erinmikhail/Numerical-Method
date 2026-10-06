import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from tests_2_1 import get_equations


def simple_iteration_method(f, df, x0, eps=1e-6, max_iter=1000):
    df0 = df(x0)
    if abs(df0) < 1e-12:
        raise ValueError("Производная f'(x0) близка к 0. Измените x0.")

    gamma = 1.0 / df0
    x_curr = x0
    errors = []

    for k in range(1, max_iter + 1):
        x_next = x_curr - gamma * f(x_curr)
        err = abs(x_next - x_curr)
        errors.append(err)

        if err < eps or abs(f(x_next)) < eps:
            return x_next, k, errors

        x_curr = x_next

    raise RuntimeError("Метод простой итерации не сошелся.")


def newton_method(f, df, x0, eps=1e-6, max_iter=100):
    x_curr = x0
    errors = []

    for k in range(1, max_iter + 1):
        df_val = df(x_curr)
        if abs(df_val) < 1e-12:
            raise ValueError(f"Производная f'(x) близка к 0 на итерации {k}.")

        x_next = x_curr - f(x_curr) / df_val
        err = abs(x_next - x_curr)
        errors.append(err)

        if err < eps or abs(f(x_next)) < eps:
            return x_next, k, errors

        x_curr = x_next

    raise RuntimeError("Метод Ньютона не сошелся.")


def main():
    x_sym, equations = get_equations()

    try:
        variant = int(input("Введите номер варианта (1-30): "))
        if variant not in equations:
            print(f"Ошибка: вариант {variant} не существует.")
            return
    except ValueError:
        print("Ошибка: введено нечисловое значение.")
        return

    eq = equations[variant]
    diff_eq = sp.diff(eq, x_sym)

    f = sp.lambdify(x_sym, eq, 'numpy')
    df = sp.lambdify(x_sym, diff_eq, 'numpy')

    x0 = 1.0
    eps = 1e-6
    a, b = 0.01, 3.0

    print(f"\nВАРИАНТ {variant}")
    print(f"Уравнение: f(x) = {eq} = 0")
    print(f"Начальное приближение x0 = {x0}, eps = {eps}\n")

    # Решение методами
    try:
        root_n, k_n, errs_n = newton_method(f, df, x0, eps)
        print(f"[Метод Ньютона]\n  Корень: {root_n:.8f}\n  Итераций: {k_n}\n  f(x) = {f(root_n):.2e}\n")
    except Exception as e:
        print(f"[Метод Ньютона] Ошибка: {e}\n")
        root_n, k_n, errs_n = None, 0, []

    try:
        root_si, k_si, errs_si = simple_iteration_method(f, df, x0, eps)
        print(f"[Простая итерация]\n  Корень: {root_si:.8f}\n  Итераций: {k_si}\n  f(x) = {f(root_si):.2e}\n")
    except Exception as e:
        print(f"[Простая итерация] Ошибка: {e}\n")
        root_si, k_si, errs_si = None, 0, []

    # Отрисовка и сохранение графика
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    x_vals = np.linspace(a, b, 500)
    with np.errstate(all='ignore'):
        y_vals = f(x_vals)

    ax1.plot(x_vals, y_vals, 'b-', label='f(x)')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    if root_n is not None:
        ax1.plot(root_n, f(root_n), 'ro', markersize=8, label=f'Корень ≈ {root_n:.4f}')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title(f"Вариант {variant}: f(x) = {eq}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()

    if errs_si:
        ax2.plot(range(1, k_si + 1), errs_si, 'r-o', label=f'Простая итерация ({k_si} итер.)')
    if errs_n:
        ax2.plot(range(1, k_n + 1), errs_n, 'g-s', label=f'Ньютон ({k_n} итер.)')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.set_title("Скорость сходимости (погрешность |x_k - x_{k-1}|)")
    ax2.set_xlabel("Номер итерации (k)")
    ax2.set_ylabel("Погрешность")
    ax2.legend()

    plt.tight_layout()
    filename = f"lab2_1_variant_{variant}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"График сохранен в файл: {filename}")


if __name__ == "__main__":
    main()