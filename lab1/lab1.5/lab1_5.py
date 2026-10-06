import math
import sys
from typing import List, Tuple, Union
from tests_5 import TESTS


def copy_matrix(A: List[List[float]]) -> List[List[float]]:
    return [row[:] for row in A]


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:

    n = len(A)
    m = len(B[0])
    k_len = len(B)
    C = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(k_len))
    return C


def qr_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:

    n = len(A)
    R = copy_matrix(A)
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n - 1):
        x = [R[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(val ** 2 for val in x))

        if norm_x < 1e-15:
            continue

        alpha = -norm_x if x[0] >= 0 else norm_x
        v = [0.0] * len(x)
        v[0] = x[0] - alpha
        for i in range(1, len(x)):
            v[i] = x[i]

        norm_v = math.sqrt(sum(val ** 2 for val in v))
        if norm_v < 1e-15:
            continue

        u = [val / norm_v for val in v]

        for j in range(k, n):
            dot = sum(u[i - k] * R[i][j] for i in range(k, n))
            for i in range(k, n):
                R[i][j] -= 2.0 * u[i - k] * dot

        for i in range(n):
            dot = sum(Q[i][j] * u[j - k] for j in range(k, n))
            for j in range(k, n):
                Q[i][j] -= 2.0 * dot * u[j - k]

    return Q, R


def extract_eigenvalues(A: List[List[float]], eps: float = 0.01) -> List[Union[float, complex]]:

    n = len(A)
    eigenvalues = []
    i = 0

    while i < n:
        if i == n - 1 or abs(A[i + 1][i]) < eps:
            eigenvalues.append(A[i][i])
            i += 1
        else:
            a = A[i][i]
            b = A[i][i + 1]
            c = A[i + 1][i]
            d = A[i + 1][i + 1]

            trace = a + d
            det = a * d - b * c
            disc = trace ** 2 - 4.0 * det

            if disc >= 0:
                eigenvalues.append((trace + math.sqrt(disc)) / 2.0)
                eigenvalues.append((trace - math.sqrt(disc)) / 2.0)
            else:
                real_part = trace / 2.0
                imag_part = math.sqrt(-disc) / 2.0
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))
            i += 2

    return eigenvalues


def qr_algorithm(
    A: List[List[float]], eps: float = 0.01, max_iter: int = 1000
) -> Tuple[List[Union[float, complex]], int, List[List[float]]]:

    Ak = copy_matrix(A)
    iters = 0

    for _ in range(max_iter):
        Q, R = qr_decomposition(Ak)
        Ak_next = matrix_multiply(R, Q)

        iters += 1
        Ak = Ak_next

    eigenvalues = extract_eigenvalues(Ak, eps=eps)
    return eigenvalues, iters, Ak


def print_matrix(name: str, M: List[List[float]]) -> None:
    print(f"\nМатрица {name}:")
    for row in M:
        print("   [ " + " ".join(f"{val:10.4f}" for val in row) + " ]")


def run_test(test_name: str, eps: float = 0.01) -> None:
    data = TESTS[test_name]
    A = data["A"]
    print_matrix("A (Исходная)", A)

    Q, R = qr_decomposition(A)
    print_matrix("Q (Ортогональная)", Q)
    print_matrix("R (Верхняя треугольная)", R)

    QR = matrix_multiply(Q, R)
    print_matrix("Q · R (Проверка восстановленной матрицы)", QR)

    max_diff = max(
        abs(A[i][j] - QR[i][j]) for i in range(len(A)) for j in range(len(A[0]))
    )
    print(f"\nМаксимальное отклонение ||A - Q·R||: {max_diff:.2e}")

    eigenvalues, iters, _ = qr_algorithm(A, eps=eps)

    print(f"\nНайденные собственные значения (с точностью ε = {eps}, за {iters} итераций):")
    for i, val in enumerate(eigenvalues):
        if isinstance(val, complex):
            sign = "+" if val.imag >= 0 else "-"
            print(f"   λ_{i+1} = {val.real:10.4f} {sign} {abs(val.imag):.4f}i")
        else:
            print(f"   λ_{i+1} = {val:10.4f}")

    print("_" * 70 + "\n")


def complex_eigenvalue_example() -> None:

    A_complex = [
        [2.0, -3.0, 1.0],
        [3.0, 2.0, -2.0],
        [0.0, 0.0, 5.0]
    ]

    print_matrix("A (Тестовая матрица с комплексными корнями)", A_complex)

    eigs_own, iters, _ = qr_algorithm(A_complex, eps=0.01)

    print(f"\n1. Собственные значения (Наша реализация QR-алгоритма, {iters} итераций):")
    for i, val in enumerate(eigs_own):
        if isinstance(val, complex):
            sign = "+" if val.imag >= 0 else "-"
            print(f"   λ_{i+1} = {val.real:10.4f} {sign} {abs(val.imag):.4f}i")
        else:
            print(f"   λ_{i+1} = {val:10.4f}")

    try:
        import numpy as np

        eigs_np = np.linalg.eigvals(A_complex)
        print("\n2. Эталонные собственные значения (Сверка через NumPy):")
        for i, val in enumerate(eigs_np):
            if np.iscomplex(val):
                sign = "+" if val.imag >= 0 else "-"
                print(f"   λ_{i+1} = {val.real:10.4f} {sign} {abs(val.imag):.4f}i")
            else:
                print(f"   λ_{i+1} = {val.real:10.4f}")
    except ImportError:
        print("\nNumPy не установлен, проверка пропущена.")

    print("_" * 70 + "\n")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    if arg == "complex":
        complex_eigenvalue_example()
        return

    if arg.isdigit():
        arg = f"test_{arg}"

    if arg == "all":
        for test_name in TESTS:
            run_test(test_name, eps=eps)
        complex_eigenvalue_example()
    elif arg in TESTS:
        run_test(arg, eps=eps)
    else:
        print(f"Тест '{arg}' не найден!")


if __name__ == "__main__":
    main()