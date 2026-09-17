import sys
from typing import List, Tuple
from tests_2 import TESTS


def check_diag(a: List[float], b: List[float], c: List[float]) -> bool:
    #модуль элемента глав диг должен быть не меньше суммы модулей бок эл
    n = len(b)
    has_strick = False
    for i in range(n):
        ai = abs(a[i]) if i > 0 else 0.0
        ci = abs(c[i]) if i < n - 1 else 0.0 
        bi = abs(b[i])

        if bi < ai + ci:
            return False
        if bi > ai + ci:
            has_strick = True
    
    return has_strick


def tridiag_solve(a: List[float], b: List[float], c: List[float], d: List[float]) -> Tuple[List[float], List[float], List[float]]:
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n
    x = [0.0] * n
    # прямой ход
    if abs(b[0]) < 1e-12:
        raise ZeroDivisionError("Деление на ноль\n")
    
    P[0] = -c[0]/ b[0]
    Q[0] = d[0]/ b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if abs(denom) < 1e-12:
            raise ZeroDivisionError("Деление на ноль на шаге i={i+1}\n")
        
        P[i] = -c[i] / denom if i < n -1 else 0.0
        Q[i] =  (d[i] - a[i] * Q[i - 1]) / denom
    
    # обратный ход
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x 


def compute_residual(a: List[float], b: List[float], c: List[float], d: List[float], x: List[float]) -> List[float]:
    n = len(b)
    res = [0.0] * n
    #вектор сравнения
    res[0] = b[0] * x[0] + c[0] * x[1] - d[0]
    for i in range(1, n - 1):
        res[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1] - d[i]
    res[n - 1] = a[n - 1] * x[n - 2] + b[n - 1] * x[n - 1] - d[n - 1]

    return res


def run_test(test_name: str) -> None:
    data = TESTS[test_name]
    a, b, c, d = data["a"], data["b"], data["c"], data["d"]
    n = len(b)

    print(f"Размерность: N = {n}")

    if not check_diag(a, b, c):
        print("Условие диагонального преобладания не выполняется.")

    try:
        P, Q, x = tridiag_solve(a, b, c, d)

        print("\n1. Прогоночные коэффициенты:")
        print(f"{'i':>4} | {'P_i':>15} | {'Q_i':>15}")
        print("-" * 40)
        for i in range(n):
            print(f"{i+1:4d} | {P[i]:15.6f} | {Q[i]:15.6f}")

        print("\n Решение системы (x):")
        for i, val in enumerate(x):
            print(f"  x[{i+1}] = {val:12.6f}")

        res = compute_residual(a, b, c, d, x)
        print(f"\n Максимальная срав ||A*x - d||: {max(abs(r) for r in res):.2e}")

    except Exception as err:
        print(f"\n Ошибка вычислений: {err}")



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
        print(f"Тест '{arg}' не найден в tests_2.py!")


if __name__ == "__main__":
    main()
