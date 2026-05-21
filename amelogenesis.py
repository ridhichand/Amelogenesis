import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# discretized s positions
s_vals = np.place(1, 5, 9)

# constant secretion speeds
vn = 1.0   # normal velocity
vt = 0.4   # tangential velocity

# front propagation speeds
cf = 1.0   # ON-front speed
cg = 2.0   # OFF-front speed

# total evaluation time

T = 10

t_eval = np.space(0, T, 1000)

def ton(s):
    return s / cf

def toff(s):
    tau = 2.0
    return ton(s) + tau
    # added constant active duration to make sure toff > ton

def active(s, t):
    return ton(s) <= t <= toff(s)

def ode_system(t, state, s):

    if active(s, t):
        dxdt = vt
        dydt = vn
    else:
        dxdt = 0.0
        dydt = 0.0

    return [dxdt, dydt]

solutions = []

for s in s_vals:

    # initial condition
    # starts on DEJ at x=s, y=0
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

# PRINT TEST INFORMATION
# ---------------------------------------------------

print("ON/OFF TIMES")
print("-" * 30)

for s, sol in solutions:
    print(
        f"s = {s:.1f}, "
        f"ton = {ton(s):.2f}, "
        f"toff = {toff(s):.2f}"
    )

plt.figure(figsize=(10, 6))

for i, s in enumerate(s_vals):

    sol = solutions[i]

    # y-position = enamel thickness
    plt.plot(
        sol.t,
        sol.y[1],
        label=f"s={s:.1f}"
    )

plt.xlabel("time")
plt.ylabel("normal displacement / enamel thickness")
plt.title("Step-Function Enamel Growth Solutions")
plt.grid(True)
plt.legend()

plt.show()
