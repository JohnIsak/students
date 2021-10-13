import numpy as np
from scipy.integrate import odeint
import timeit
import numba


def f(u, t, sigma, rho, beta):
    x, y, z = u
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


u0 = [1.0, 0.0, 0.0]
tspan = (0., 100.)
t = np.linspace(0, 100, 1001)
sol = odeint(f, u0, t, args=(10.0, 28.0, 8 / 3))


def time_func():
    odeint(f, u0, t, args=(10.0, 28.0, 8 / 3), rtol=1e-8, atol=1e-8)


time_func()
asd = timeit.Timer(time_func).timeit(number=100) / 100  # 0.07502969299999997 seconds
print(asd)

numba_f = numba.jit(f, nopython=True)


def time_func():
    odeint(numba_f, u0, t, args=(10.0, 28.0, 8 / 3), rtol=1e-8, atol=1e-8)


time_func()
qwe = timeit.Timer(time_func).timeit(number=100) / 100  # 0.04437951900000001 seconds
print(qwe)
True