import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# discretized s positions
s_vals = jnp.linspace(1, 5, 100)

# normal and tangential velocities
def vn(s, t):
    return 1.0 + 0.2 * jnp.sin(s)

def vt(s, t):
    return 0.4 + 0.1 * jnp.cos(t)

# front propagation speed
cf = 1.0

def phi_f(t):      # ON-front
    return cf * t

def Delta(t):
    return 2.0 + 0.5 * jnp.sin(t)

def phi_g(t):
    return phi_f(t) - Delta(t)

# total simulation time
T = 10
t_eval = np.linspace(0, T, 1000)

def active(s, t):
    return (phi_g(t) <= s) & (s <= phi_f(t))

# ODE system
def ode_system(t, state, s):

    x, y = state

    if bool(active(s, t)):
        dxdt = float(vt(s, t))
        dydt = float(vn(s, t))
    else:
        dxdt = 0.0
        dydt = 0.0

    return [dxdt, dydt]

# Solve ODEs

solutions = []
termination_times = []

for s in s_vals:

    s = float(s)

    y0 = [s, 0]

    sol = solve_ivp(
        ode_system,
        [0, T],
        y0,
        t_eval=t_eval,
        args=(s,),
        method="RK45"
    )

    solutions.append(sol)

    result = root_scalar(
        lambda t: phi_g(t) - s,
        bracket=[0, T]
    )

    termination_times.append(result.root)

# Final OES geometry

x_final = []
y_final = []

for sol in solutions:
    x_final.append(sol.y[0][-1])
    y_final.append(sol.y[1][-1])

x_final = np.array(x_final)
y_final = np.array(y_final)

# Compute OES arc length S(s)

S = [0.0]

for i in range(1, len(x_final)):

    dx = x_final[i] - x_final[i-1]
    dy = y_final[i] - y_final[i-1]

    dS = np.sqrt(dx**2 + dy**2)

    S.append(S[-1] + dS)

S = np.array(S)

# Compute dt*/ds

termination_times = np.array(termination_times)

dt_ds = np.gradient(
    termination_times,
    np.array(s_vals)
)

# Divide OES into intervals

N = 20

edges = np.linspace(0, S[-1], N + 1)

T_interval = []

for k in range(N):

    mask = (S >= edges[k]) & (S < edges[k+1])

    developmental_time = np.sum(
        np.abs(dt_ds[mask])
    )

    T_interval.append(developmental_time)

T_interval = np.array(T_interval)

# Convert to Perikymata

R = 8      # Retzius periodicity (days)

PK = 365 * T_interval / R

# Growth over time Plot

plt.figure(figsize=(10,6))

for sol in solutions:
    plt.plot(sol.t, sol.y[1])

plt.xlabel("Time")
plt.ylabel("Normal displacement / enamel thickness")
plt.title("Step-Function Enamel Growth")
plt.grid(True)

# Final OES Plot

plt.figure(figsize=(8,6))

plt.plot(x_final, y_final, "-", linewidth=2)

plt.xlabel("Tangential Position")
plt.ylabel("Enamel Thickness")
plt.title("Final Outer Enamel Surface")
plt.grid(True)

# Perikymata Distribution Plot

centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(8,5))

plt.plot(centers, PK, "o-")

plt.xlabel("Distance Along OES")
plt.ylabel("Perikymata")
plt.title("Perikymata Distribution")
plt.grid(True)

plt.show()
