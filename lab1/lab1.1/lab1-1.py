import  sys
from tests import TESTS

#матричные операции 
def mat_mult(A,B):
    #C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + ... + A[i][k-1]*B[k-1][j]
    n = len(A)
    m = len(B[0])
    k_len = len(B)

    C = [[0.0]*m for i in range(n)]

    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(k_len))
    return C


def mat_vec_mult(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def compute_determinant(U, num_swaps):
    det = (-1.0) ** num_swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det


def compute_inverse(L, U, P):
    n = len(L)
    A_inv = [[0.0] * n for _ in range(n)]

    for k in range(n):
        e_k = [1.0 if i == k else 0.0 for i in range(n)]
        x_k = solve_lu(L, U, P, e_k)
        for i in range(n):
            A_inv[i][k] = x_k[i]

    return A_inv


#LUP разложение 
def lup_decompose(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    num_swaps = 0  # счетчик перестановок строк

    for k in range(n):
        pivot_row = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_row = i

        if max_val < 1e-12:
            raise ValueError("Матрица вырождена")
        
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]
            num_swaps += 1

        L[k][k] = 1.0

        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            U[i][k] = 0.0

            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]
                    
    return L, U, P, num_swaps


#Слфн реш
def solve_lu(L, U, P, b):
    n = len(L)
    Pb = mat_vec_mult(P, b)

    y = [0.0] * n #прямой ход L*y=Pb
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = Pb[i] - s
    
    x = [0.0] * n #обратный ход U*x=y
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    
    return x


#формат вывод
def print_matrix(matrix, prec=4):
    for row in matrix:
        formatted_row = [f"{val:10.{prec}f}" for val in row]
        print("  [" + " ".join(formatted_row) + " ]")


def print_vector(vector, prec=4):
    formatted_vector = [f"{val:10.{prec}f}" for val in vector]
    print("  [" + " ".join(formatted_vector) + " ]")


# вывод Требований
def run_test(test_name):
    print("Начало\n")    
    A = TESTS[test_name]["A"]
    b = TESTS[test_name]["b"]

    print("\nИсходная матрица A:")
    print_matrix(A, prec=2)
    print("\nПравая часть b:", b)

    # Разложение
    L, U, P, num_swaps = lup_decompose(A)

    print("1.\n")
    print("Матрица L:")
    print_matrix(L)
    print("\nМатрица U:")
    print_matrix(U)


    LU = mat_mult(L, U)
    print("2. рез * L · U:\n")
    print_matrix(LU)


    x = solve_lu(L, U, P, b)
    print("3. реш сис Ax = b:\n")
    print_vector(x, prec=6)


    A_inv = compute_inverse(L, U, P)
    print("4. A в -1 степени, найденная через LU-разложение:\n")
    print_matrix(A_inv)


    det = compute_determinant(U, num_swaps)
    print(f"5. det A, через LU-разлож:\n")
    print(f"  det(A) = {det:.6f}")


    A_Ainv = mat_mult(A, A_inv)
    print("6. ПРОВЕРКА A · A в -1 = E :\n")
    print_matrix(A_Ainv, prec=4)


    PA = mat_mult(P, A)
    print("7.  L · U = P · A:\n")
    print("Матрица P · A:")
    print_matrix(PA)
    
    is_valid = all(
        abs(LU[i][j] - PA[i][j]) < 1e-9
        for i in range(len(A))
        for j in range(len(A))
    )
    print(f"\n  Совпадение L · U и P · A: {'Элементы равны - успух' if is_valid else 'error'}")


if __name__ == "__main__":
    target_test = sys.argv[1] if len(sys.argv) > 1 else "test_1"
    if target_test in TESTS:
        run_test(target_test)
    else:
        print(f"Тест '{target_test}' не найден!")
        print(f"Доступные варианты: {list(TESTS.keys())}")