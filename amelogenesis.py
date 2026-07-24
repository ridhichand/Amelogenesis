import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# parameters

# discretized EDJ arc length
s_vals = jnp.linspace(1,5,500)

T = 10
t_eval = np.linspace(0,T,1000)

cf = 1.0               # ON-front speed

# velocities

def vn(s,t):
    return 1.0 + 0.2*jnp.sin(s)

def vt(s,t):
    return 0.4 + 0.1*jnp.cos(t)

# front motion

def phi_f(t):
    return cf*t

def Delta(t):
    return 2.0 + 0.5*jnp.sin(t)

def phi_g(t):
    return phi_f(t)-Delta(t)

# smoother active window

def smooth_active(s,t,beta=20):

    left = 0.5*(1+jnp.tanh(beta*(s-phi_g(t))))
    right = 0.5*(1+jnp.tanh(beta*(phi_f(t)-s)))

    return left*right

# ode

def ode_system(t,state,s):

    x,y = state

    a = float(smooth_active(s,t))

    dxdt = a*float(vt(s,t))
    dydt = a*float(vn(s,t))

    return [dxdt,dydt]

# solve

solutions = []
termination_times = []

for s in s_vals:

    s = float(s)

    y0 = [s,0]

    sol = solve_ivp(
        ode_system,
        [0,T],
        y0,
        t_eval=t_eval,
        args=(s,),
        method="RK45",
        max_step=0.05 

    solutions.append(sol)

    result = root_scalar(
        lambda t: phi_g(t)-s,
        bracket=[0,T]
    )

    termination_times.append(result.root)

termination_times = np.array(termination_times)

# final geometry

x_final = np.array([sol.y[0,-1] for sol in solutions])
y_final = np.array([sol.y[1,-1] for sol in solutions])

# outer enamel surface arc length

S = [0]

for i in range(1,len(x_final)):

    dx = x_final[i]-x_final[i-1]
    dy = y_final[i]-y_final[i-1]

    S.append(S[-1]+np.sqrt(dx**2+dy**2))

S = np.array(S)

# perikymata distribution

dt_ds = np.gradient(
    termination_times,
    np.array(s_vals)
ds = float(s_vals[1] - s_vals[0])

N = 20

edges = np.linspace(0,S[-1],N+1)

T_interval = []

for k in range(N):

    mask = (S>=edges[k])&(S<edges[k+1])

    T_interval.append(
        np.sum(np.abs(dt_ds[mask])) * ds
    )

T_interval = np.array(T_interval)

edges = np.linspace(0,S[-1],N+1)

T_interval = []

R = 8

PK = 365*T_interval/R

# diagnostic functions

def plot_front_motion():

    t = np.linspace(0,T,500)

    plt.figure(figsize=(7,5))

    plt.plot(t,[phi_f(tt) for tt in t],label="ON Front")
    plt.plot(t,[phi_g(tt) for tt in t],label="OFF Front")

    plt.xlabel("Time")
    plt.ylabel("Front Position")
    plt.title("Front Motion")

    plt.legend()
    plt.grid(True)

def plot_kymograph():

    t = np.linspace(0,T,300)
    s = np.linspace(1,5,200)

    A = np.zeros((len(s),len(t)))

    for i,ss in enumerate(s):
        for j,tt in enumerate(t):

            A[i,j]=float(
                smooth_active(ss,tt)
            )

    plt.figure(figsize=(8,5))

    plt.imshow(
        A,
        extent=[0,T,1,5],
        origin="lower",
        aspect="auto"
    )

    plt.xlabel("Time")
    plt.ylabel("EDJ Arc Length")
    plt.title("Activity Kymograph")

    plt.colorbar(label="Activity")

def plot_geometry(times):

    plt.figure(figsize=(8,6))

    for t in times:

        x=[]
        y=[]

        for sol in solutions:

            idx=np.argmin(np.abs(sol.t-t))

            x.append(sol.y[0,idx])
            y.append(sol.y[1,idx])

        plt.plot(x,y,label=f"t={t}")

    plt.xlabel("Tangential Position")
    plt.ylabel("Enamel Thickness")
    plt.title("Geometry Evolution")

    plt.legend()
    plt.grid(True)

def plot_quiver(time):

    x=[]
    y=[]
    u=[]
    v=[]

    for s,sol in zip(s_vals,solutions):

        s=float(s)

        idx=np.argmin(np.abs(sol.t-time))

        xx=sol.y[0,idx]
        yy=sol.y[1,idx]

        x.append(xx)
        y.append(yy)

        a=float(smooth_active(s,time))

        u.append(a*float(vt(s,time)))
        v.append(a*float(vn(s,time)))

    plt.figure(figsize=(8,6))

    plt.plot(x,y,'k-')

    plt.quiver(
        x,
        y,
        u,
        v,
        angles='xy',
        scale_units='xy',
        scale=1
    )

    plt.xlabel("Tangential Position")
    plt.ylabel("Enamel Thickness")
    plt.title(f"Velocity Field at t={time}")

    plt.grid(True)

# original plots

plt.figure(figsize=(10,6))

for sol in solutions:
    plt.plot(sol.t,sol.y[1])

plt.xlabel("Time")
plt.ylabel("Normal Displacement")
plt.title("Enamel Growth")
plt.grid(True)

plt.figure(figsize=(8,6))

plt.plot(x_final,y_final,'-',linewidth=2)

plt.xlabel("Tangential Position")
plt.ylabel("Enamel Thickness")
plt.title("Final Outer Enamel Surface")
plt.grid(True)

centers=0.5*(edges[:-1]+edges[1:])

plt.figure(figsize=(8,5))

plt.plot(centers,PK,'o-')

plt.xlabel("Distance Along OES")
plt.ylabel("Perikymata")
plt.title("Perikymata Distribution")
plt.grid(True)

# diagnostic plots

plot_front_motion()

plot_kymograph()

plot_geometry([2,4,6,8,10])

plot_quiver(6)

plt.show()
