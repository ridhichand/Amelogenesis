import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# discretized s positions
s_vals = jnp.linspace(1, 5, 100)

# normal and tangential velocities
def vn(s, t):
    return 1.0 + 0.2 * jnp.sin(s)

def vt(s, t):
    return 0.4 + 0.1 * jnp.cos(t)

# front propagation speeds
cf = 1.0   

def phi_f(t): # ON-front speed
    return cf * t

def Delta(t):
    return 2.0 + 0.5 * jnp.sin(t)

def phi_g(t): # OFF-front speed
    return phi_f(t) - Delta(t)

# total evaluation time

T = 10

t_eval = np.linspace(0, T, 1000)

def active(s, t):
    return (phi_g(t) <= s) & (s <= phi_f(t))

def ode_system(t, state, s):

    x, y = state
    if bool(active(s, t)):
        dxdt = float(vt(s, t))
        dydt = float(vn(s, t))
    else:
        dxdt = 0.0
        dydt = 0.0

    return [dxdt, dydt]

solutions = []

for s in s_vals:

    # initial condition
    # starts on DEJ at x=s, y=0
    s = float(s)
    y0 = [s, 0]

    sol = solve_ivp(
        ode_system,
        [0, T],
        y0,
        t_eval=t_eval,
        args=(s,),
        method='RK45'
    )

    solutions.append(sol)

plt.figure(figsize=(10, 6))

for s, sol in zip(s_vals, solutions):
    plt.plot(
        sol.t,
        sol.y[1]
    )

plt.xlabel("time")
plt.ylabel("normal displacement / enamel thickness")
plt.title("Step-Function Enamel Growth Solutions")
plt.grid(True)

plt.show()
