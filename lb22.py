import json
import math
import sys


A = 2.0


def F(x1, x2):
    return x1*x1/(A*A) + x2*x2/((A/2)**2) - 1, A*x2 - math.exp(x1) - x1


def jacobian(x1, x2):
    # Матрица Якоби системы F1=0, F2=0
    return [[2*x1/(A*A), 2*x2/((A/2)**2)],
            [-math.exp(x1)-1, A]]


def solve_2x2(matrix, rhs):
    a, b = matrix[0]
    c, d = matrix[1]
    det = a*d - b*c
    if abs(det) < 1e-14:
        raise ZeroDivisionError("Якобиан вырожден или почти вырожден")
    r1, r2 = rhs
    return ((r1*d - b*r2)/det, (a*r2 - r1*c)/det)


def norm_inf(v):
    return max(abs(v[0]), abs(v[1]))


def simple_iteration(x1, x2, eps, max_iter, alpha, beta):
    # Итерационная форма x_new = x - D*F(x).
    # Малые alpha, beta нужны для устойчивой сходимости.
    rows = []
    for k in range(1, max_iter + 1):
        f1, f2 = F(x1, x2)
        nx1 = x1 - alpha*f1
        nx2 = x2 - beta*f2
        delta = max(abs(nx1-x1), abs(nx2-x2))
        x1, x2 = nx1, nx2
        residual = norm_inf(F(x1, x2))
        rows.append((k, x1, x2, residual, delta))
        if delta < eps and residual < eps:
            return (x1, x2), rows
    raise RuntimeError("Простая итерация не сошлась. Попробуйте меньшие alpha/beta или другое начальное приближение.")


def newton(x1, x2, eps, max_iter):
    rows = []
    for k in range(1, max_iter + 1):
        values = F(x1, x2)
        delta = solve_2x2(jacobian(x1, x2), values)
        nx1 = x1 - delta[0]
        nx2 = x2 - delta[1]
        step = max(abs(nx1-x1), abs(nx2-x2))
        x1, x2 = nx1, nx2
        residual = norm_inf(F(x1, x2))
        rows.append((k, x1, x2, residual, step))
        if step < eps and residual < eps:
            return (x1, x2), rows
    raise RuntimeError("Метод Ньютона не сошелся за max_iter итераций")


def print_result(title, result):
    (x1, x2), rows = result
    print(f"\n{title}: x1={x1:.12f}, x2={x2:.12f}; ||F||inf={norm_inf(F(x1,x2)):.3e}; итераций={len(rows)}")
    print(" k          x1                  x2              ||F(x_k)||inf       шаг")
    for k, u, v, residual, step in rows:
        print(f"{k:2d}  {u: .12f}    {v: .12f}      {residual:.3e}      {step:.3e}")


def main():
    if len(sys.argv) != 2:
        print("Запуск: python3 lb22.py data_lb22.json")
        raise SystemExit(1)
    with open(sys.argv[1], encoding="utf-8") as file:
        data = json.load(file)
    eps = float(data["epsilon"])
    x1 = float(data["x1_0"])
    x2 = float(data["x2_0"])
    max_iter = int(data.get("max_iter", 10000))
    alpha = float(data.get("alpha", 0.05))
    beta = float(data.get("beta", 0.05))
    if eps <= 0 or alpha <= 0 or beta <= 0:
        raise ValueError("epsilon, alpha и beta должны быть положительными")
    print("Система при a=2:")
    print("F1=x1^2/4 + x2^2 - 1 = 0")
    print("F2=2*x2 - exp(x1) - x1 = 0")
    print(f"epsilon={eps}; start=({x1}, {x2}); alpha={alpha}; beta={beta}")
    print_result("Простая итерация", simple_iteration(x1, x2, eps, max_iter, alpha, beta))
    print_result("Ньютон", newton(x1, x2, eps, max_iter))


if __name__ == "__main__":
    main()
