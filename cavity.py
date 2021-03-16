import numpy as np
import matplotlib.pyplot as plt

Nx = 100
Ny = 100

u = np.loadtxt("u.txt")
v = np.loadtxt("v.txt")
p = np.loadtxt("p.txt")

xf = np.linspace(0, 1, Nx+1)
yf = np.linspace(0, 1, Ny+1)
xc = np.zeros(Nx+2)
yc = np.zeros(Ny+2)
xc[1:-1] = (xf[1:] + xf[:-1]) / 2
yc[1:-1] = (yf[1:] + yf[:-1]) / 2
xc[0] = 2*xf[0] - xc[1]
xc[-1] = 2*xf[-1] - xc[-2]
yc[0] = 2*yf[0] - yc[1]
yc[-1] = 2*yf[-1] - yc[-2]

#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3, 9))

ax1.set_title('u')
x, y = np.meshgrid(xf, yc)
C1 = ax1.contourf(x, y, u.T, 100)
ax1.set_aspect('equal', 'box')
fig.colorbar(C1, ax=ax1)

ax2.set_title('v')
x, y = np.meshgrid(xc, yf)
C2 = ax2.contourf(x, y, v.T, 100)
ax2.set_aspect('equal', 'box')
fig.colorbar(C2, ax=ax2)

ax3.set_title('Pressure')
x, y = np.meshgrid(xc, yc)
C3 = ax3.contourf(x, y, p.T, 100)
ax3.set_aspect('equal', 'box')
fig.colorbar(C3, ax=ax3)

fig.tight_layout()

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

uc = u[Nx//2, :]
vc = v[:, Ny//2]

ax.plot(2*xc[1:-1]-1, -vc[-1:1:-1]/0.6)
ax.plot(uc[1:-1], 2*yc[1:-1]-1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', 'box')
ax.grid(True)

plt.show()
