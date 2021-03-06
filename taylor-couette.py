import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

Nx = 100
Ny = 100
L = 3.5
ri = 0.25
ro = 1.5
omega = 1

u = np.loadtxt("u.txt")
v = np.loadtxt("v.txt")
p = np.loadtxt("p.txt")

xf = np.linspace(0, L, Nx+1)
yf = np.linspace(0, L, Ny+1)
xc = np.zeros(Nx+2)
yc = np.zeros(Ny+2)
xc[1:-1] = (xf[1:] + xf[:-1]) / 2
yc[1:-1] = (yf[1:] + yf[:-1]) / 2
xc[0] = 2*xf[0] - xc[1]
xc[-1] = 2*xf[-1] - xc[-2]
yc[0] = 2*yf[0] - yc[1]
yc[-1] = 2*yf[-1] - yc[-2]

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

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
fig, ax = plt.subplots(figsize=(6, 6))

ut = (u[1:, 1:-1] + u[:-1, 1:-1]) / 2
vt = (v[1:-1, 1:] + v[1:-1, :-1]) / 2
x, y = np.meshgrid(xc[1:-1], yc[1:-1])

wi = Wedge((L/2, L/2), ri, 0, 360, color='grey', zorder=10)
wo = Wedge((L/2, L/2), ro+1, 0, 360, width=1, color='grey', zorder=10)

ax.streamplot(x, y, ut.T, vt.T, color='k')
ax.add_patch(wi)
ax.add_patch(wo)
ax.set_aspect('equal', 'box')

fig.tight_layout()

#%%
fig, ax = plt.subplots()

r = np.linspace(ri, ro, 21)
ut = 1/(ro**2-ri**2) * (-omega*ri**2*r + omega*ri**2*ro**2/r)

x = np.hstack((L/2-r, L/2+r))
vt = np.hstack((ut, -ut))

ax.plot(xc, v[:, Ny//2])
ax.plot(x, vt, 'ro', markersize=3)
ax.legend(['IBM', 'Theory'])

fig.tight_layout()

plt.show()
