import math
import sys
from typing import List, Tuple
from tests_4 import TESTS


def is_symmetric(A: List[List[float]], tol: float = 1e-8) -> bool:
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def copy_matrix(A: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in A]


def create_1_matrix(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def find_max_of_diag(A:  List[List[float]]) -> Tuple[int, int, float]:
    n = len(A)
    p, q = 0, 1
    max_val = abs(A[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    
    return p, q, max_val


def jacobi_rotation(A_in: List[List[float]], eps: float = 1e-8, max_iter: int = 10000) -> Tuple[List[float], List[List[float]], int]:
    n = len(A_in)
    A = copy_matrix(A_in)
    V = create_1_matrix(n)
    iter = 0

    while iter < max_iter:
        p, q, max_val = find_max_of_diag(A)

        if max_val < eps:
            break

        if abs(A[p][p] - A[q][q]) < eps:
            ph = math.pi / 4
        else:
            ph = 0.5 * math.atan(2 * A[p][q] / (A[p][p] - A[q][q]))

        c = math.cos(ph)
        s = math.sin(ph)

        A_new = copy_matrix(A)

        A_new[p][p] = c**2 * A[p][p] + 2.0 * c * s * A[p][q] + s**2 * A[q][q]
        A_new[q][q] = s**2 * A[p][p] - 2.0 * c * s * A[p][q] + c**2 * A[q][q]
        A_new[p][q] = 0.0
        A_new[q][p] = 0.0

        for i in range(n):
            if i != p and i != q:
                A_new[i][p] = c * A[i][p] + s * A[i][q]
                A_new[p][i] = A_new[i][p]
                A_new[i][q] = -s * A[i][p] + c * A[i][q]
                A_new[q][i] = A_new[i][q]

        A = A_new

        for i in range(n):
            v_ip = V[i][p]
            v_iq = V[i][q]
            V[i][p] = c * v_ip + s * v_iq
            V[i][q] = -s * v_ip + c * v_iq

        iter += 1

    eigenvalues = [A[i][i] for i in range(n)]
    return eigenvalues, V, iter


def matrix_mult(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    m = len(B[0])
    k_len = len(B)
    C = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(k_len))
    return C


def create_diag_matrix(diag: List[float]) -> List[List[float]]:
    n = len(diag)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = diag[i]
    return L


def check_eigen_equation(A: List[List[float]], V: List[List[float]], lambdas: List[float]) -> Tuple[List[List[float]], List[List[float]], float]:
    L = create_diag_matrix(lambdas)
    AV = matrix_mult(A, V)
    VL = matrix_mult(V, L)

    n = len(A)
    max_diff = 0.0
    for i in range(n):
        for j in range(n):
            diff = abs(AV[i][j] - VL[i][j])
            if diff > max_diff:
                max_diff = diff

    return AV, VL, max_diff


def run_test(test_name: str, eps: float = 1e-4) -> None:
    data = TESTS[test_name]
    A = data["A"]
    n = len(A)

    print(f"Размерность матрицы: N = {n}")
    print(f"1. Заданная точность вычислений ε = {eps}")

    if not is_symmetric(A):
        print("Матрица не является симметричной! Метод Якоби неприменим.")
        return

    lambdas, V, iters = jacobi_rotation(A, eps=eps)

    print(f"\nКоличество итераций: {iters}")

    print("\n2. Найденные собственные значения (lambda):")
    for i, val in enumerate(lambdas):
        print(f"   λ_{i+1} = {val:12.6f}")

    print("\n3. Матрица собственных векторов V:")
    for i in range(n):
        row_str = " ".join(f"{V[i][j]:10.6f}" for j in range(n))
        print(f"   [ {row_str} ]")

    AV, VL, max_diff = check_eigen_equation(A, V, lambdas)

    print("\n4. Проверка равенства A · V = V · Λ:")
    print(f"   Максимальный модуль разности ||A·V - V·Λ|| = {max_diff:.2e}")

    if max_diff < eps * 10:
        print("Равенство A · V = V · Λ выполнено с высокой точностью!")
    else:
        print(" Разность превышает допустимый порог.")
    
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
        print(f"Тест '{arg}' не найден в tests_4.py!")


if __name__ == "__main__":
    main()

