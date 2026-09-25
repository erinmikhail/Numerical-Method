import copy
import sys
from tests_3 import TESTS
from typing import List, Tuple

def check_diag_dom(A: List[List[float]]) -> bool:
    """чек усл преоблад гл диг"""
    n = len(A)
    has_strict = False
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)

        if diag < off_diag_sum:
            return False
        if diag > off_diag_sum:
            has_strict = True

    return has_strict


def vector_norm_diff(x1: List[float], x2: List[float]) -> float:
    return max(abs(a - b) for a, b in zip(x1, x2))


def simple_iteration(A: List[List[float]], b: List[float], eps: float = 0.01, max_iter: int = 10000) -> Tuple[List[float], int]:

    n = len(A)
    x = [0.0] * n
    iterations = 0

    while iterations < max_iter:
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        iterations += 1
        if vector_norm_diff(x_new, x) < eps:
            return x_new, iterations

        x = x_new

    raise TimeoutError("Метод простых итераций не сошелся за отведенное число шагов.")


def seidel_iteration(A: List[List[float]], b: List[float], eps: float = 0.01, max_iter: int = 10000) -> Tuple[List[float], int]:

    n = len(A)
    x = [0.0] * n
    iterations = 0

    while iterations < max_iter:
        x_new = copy.deepcopy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        iterations += 1
        if vector_norm_diff(x_new, x) < eps:
            return x_new, iterations

        x = x_new

    raise TimeoutError("Метод Зейделя не сошелся за отведенное число шагов.")


def exact_solve_gauss(A: List[List[float]], b: List[float]) -> List[float]:
    """эталонного точного решения."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    # Прямой ход
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot

        for k in range(i + 1, n):
            factor = M[k][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))

    return x


def compute_residual(A: List[List[float]], b: List[float], x: List[float]) -> float:
    """норма невязки"""
    n = len(A)
    res = []
    for i in range(n):
        val = sum(A[i][j] * x[j] for j in range(n)) - b[i]
        res.append(abs(val))
    return max(res)


def run_test(test_name: str, eps: float = 0.01) -> None:
    data = TESTS[test_name]
    A, b = data["A"], data["b"]
    n = len(A)

    print(f"Размерность системы N: {n}")
    print(f"1. Заданная точность ε: {eps}")

    if not check_diag_dom(A):
        print("Строгое диагональное преобладание не выполняется.")

    # 2. Вычисление точного (эталонного) решения
    x_exact = exact_solve_gauss(A, b)
    print("\n3. Эталон решение:")
    for i, val in enumerate(x_exact):
        print(f"   x_exact[{i+1}] = {val:12.6f}")

    # 3. Выполнение Итерационных методов
    try:
        x_simple, iter_simple = simple_iteration(A, b, eps=eps)
        x_seidel, iter_seidel = seidel_iteration(A, b, eps=eps)

        print("\n2. Результаты итерационных методов:")
        print(f"{'Переменная':>10} | {'Простые итерации':>18} | {'Метод Зейделя':>18}")
        print("-" * 52)
        for i in range(n):
            print(f"{'x[' + str(i+1) + ']':>10} | {x_simple[i]:18.6f} | {x_seidel[i]:18.6f}")

        print("\n2. Количество итераций:")
        print(f"   • Метод простых итераций: {iter_simple} итер.")
        print(f"   • Метод Зейделя:          {iter_seidel} итер.")

        print("\n4. Сравнение методов по скорости сходимости:")
        diff_iter = iter_simple - iter_seidel
        if diff_iter > 0:
            speedup = (iter_simple / iter_seidel) if iter_seidel > 0 else 0
            print(f"   ► Метод Зейделя сошелся быстрее на {diff_iter} итер. (в {speedup:.1f} раза).")
        elif diff_iter < 0:
            print(f"   ► Метод простых итераций сошелся быстрее на {abs(diff_iter)} итер.")
        else:
            print("   ► Оба метода сошлись за одинаковое число итераций.")

        # Оценка невязок
        res_simple = compute_residual(A, b, x_simple)
        res_seidel = compute_residual(A, b, x_seidel)
        print(f"\n   Невязка (Простые итерации): {res_simple:.2e}")
        print(f"   Невязка (Зейдель):          {res_seidel:.2e}")

    except Exception as err:
        print(f"\n Ошибка вычислений: {err}")

    print("_" * 70 + "\n")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"

    if arg.isdigit():
        arg = f"test_{arg}"

    if arg == "all":
        for test_name in TESTS:
            run_test(test_name)
    elif arg in TESTS:
        run_test(arg)
    else:
        print(f"Тест '{arg}' не найден в tests_3.py!")


if __name__ == "__main__":
    main()
