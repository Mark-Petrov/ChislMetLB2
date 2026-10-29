import json
import math
import sys


def f(x):
    return math.log(x + 1) - 2 * x + 0.5


def df(x):
    return 1 / (x + 1) - 2


def simple_iteration(x0, eps, max_iter):
    # x = phi(x) = (ln(x + 1) + 0.5) / 2
    phi = lambda x: (math.log(x + 1) + 0.5) / 2
    rows = []
    x = x0
    for k in range(1, max_iter + 1):
        xn = phi(x)
        delta = abs(xn - x)
        rows.append((k, xn, abs(f(xn)), delta))
        if delta < eps and abs(f(xn)) < eps:
            return xn, rows
        x = xn
    raise RuntimeError("Метод простой итерации не сошелся за max_iter итераций")


def newton(x0, eps, max_iter):
    rows = []
    x = x0
    for k in range(1, max_iter + 1):
        d = df(x)
        if abs(d) < 1e-15:
            raise ZeroDivisionError("Производная слишком близка к нулю")
        xn = x - f(x) / d
        delta = abs(xn - x)
        rows.append((k, xn, abs(f(xn)), delta))
        if delta < eps and abs(f(xn)) < eps:
            return xn, rows
        x = xn
    raise RuntimeError("Метод Ньютона не сошелся за max_iter итераций")


def print_result(title, result):
    root, rows = result
    print(f"\n{title}: x = {root:.12f}; |f(x)| = {abs(f(root)):.3e}; итераций = {len(rows)}")
    print(" k          x_k                 |f(x_k)|            |x_k-x_(k-1)|")
    for k, x, residual, delta in rows:
        print(f"{k:2d}  {x: .12f}      {residual:.3e}       {delta:.3e}")


def main():
    if len(sys.argv) != 2:
        print("Запуск: python3 lb21.py data_lb21.json")
        raise SystemExit(1)
    with open(sys.argv[1], encoding="utf-8") as file:
        data = json.load(file)
    eps = float(data["epsilon"])
    x0 = float(data["x0"])
    max_iter = int(data.get("max_iter", 1000))
    if eps <= 0 or x0 <= -1:
        raise ValueError("epsilon должна быть положительной, x0 должен быть больше -1")
    print(f"f(x)=ln(x+1)-2x+0.5; epsilon={eps}; x0={x0}")
    print_result("Простая итерация", simple_iteration(x0, eps, max_iter))
    print_result("Ньютон", newton(x0, eps, max_iter))


if __name__ == "__main__":
    main()
