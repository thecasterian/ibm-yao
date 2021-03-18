import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

Nx = 80
Ny = 68

R = 0.5

u = np.loadtxt("u.txt")
v = np.loadtxt("v.txt")
p = np.loadtxt("p.txt")

xf = np.array([-4.1349, -3.7874, -3.4715, -3.1844, -2.9233, -2.6859, -2.4702, -2.2740, -2.0957, -1.9336, -1.7862, -1.6522, -1.5304, -1.4197, -1.3191, -1.2276, -1.1444, -1.0688, -1, -0.9375, -0.875, -0.8125, -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1, 1.0688, 1.1444, 1.2276, 1.3191, 1.4197, 1.5304, 1.6522, 1.7862, 1.9336, 2.0957, 2.2740, 2.4702, 2.6859, 2.9233, 3.1844, 3.4715, 3.7874, 4.1349, 4.5172, 4.9377, 5.4002, 5.9089, 6.4686, 7.0842, 7.7614, 8.5062, 9.3256, 10.2269, 11.2184, 12.3090])
yf = np.array([-4.1349, -3.7874, -3.4715, -3.1844, -2.9233, -2.6859, -2.4702, -2.2740, -2.0957, -1.9336, -1.7862, -1.6522, -1.5304, -1.4197, -1.3191, -1.2276, -1.1444, -1.0688, -1, -0.9375, -0.875, -0.8125, -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1, 1.0688, 1.1444, 1.2276, 1.3191, 1.4197, 1.5304, 1.6522, 1.7862, 1.9336, 2.0957, 2.2740, 2.4702, 2.6859, 2.9233, 3.1844, 3.4715, 3.7874, 4.1349])
xc = np.zeros(Nx+2)
yc = np.zeros(Ny+2)
xc[1:-1] = (xf[1:] + xf[:-1]) / 2
yc[1:-1] = (yf[1:] + yf[:-1]) / 2
xc[0] = 2*xf[0] - xc[1]
xc[-1] = 2*xf[-1] - xc[-2]
yc[0] = 2*yf[0] - yc[1]
yc[-1] = 2*yf[-1] - yc[-2]

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 3))

ax1.set_title('u')
x, y = np.meshgrid(xf, yc)
w = Wedge((0, 0), R, 0, 360, color='grey', zorder=10)
C1 = ax1.contourf(x, y, u.T, 100)
ax1.add_patch(w)
ax1.set_aspect('equal', 'box')
fig.colorbar(C1, ax=ax1)

ax2.set_title('v')
x, y = np.meshgrid(xc, yf)
w = Wedge((0, 0), R, 0, 360, color='grey', zorder=10)
ax2.add_patch(w)
C2 = ax2.contourf(x, y, v.T, 100)
ax2.set_aspect('equal', 'box')
fig.colorbar(C2, ax=ax2)

ax3.set_title('Pressure')
x, y = np.meshgrid(xc, yc)
w = Wedge((0, 0), R, 0, 360, color='grey', zorder=10)
ax3.add_patch(w)
C3 = ax3.contourf(x, y, p.T, 100)
ax3.set_aspect('equal', 'box')
fig.colorbar(C3, ax=ax3)

fig.tight_layout()

#%%
fig, ax = plt.subplots(figsize=(8, 4))

ut = (u[1:, 1:-1] + u[:-1, 1:-1]) / 2
vt = (v[1:-1, 1:] + v[1:-1, :-1]) / 2
x, y = np.meshgrid(xc[1:-1], yc[1:-1])

w = Wedge((0, 0), R, 0, 360, color='grey', zorder=10)

ax.set_title('Velocity Magnitude')
ax.add_patch(w)
C = ax.contourf(x, y, np.sqrt(ut.T**2 + vt.T**2), 100, cmap='coolwarm')
ax.set_aspect('equal', 'box')
fig.colorbar(C, ax=ax, fraction=0.0235, pad=0.04)

fig.tight_layout()

#%%
fig, ax = plt.subplots(figsize=(8, 4))

x, y = np.meshgrid(xf, yf)
w = Wedge((0, 0), R, 0, 360, color='grey', zorder=10)

ax.add_patch(w)
ax.plot(x, y, 'k', linewidth=1)
ax.plot(x.T, y.T, 'k', linewidth=1)
ax.set_aspect('equal', 'box')
ax.axis('off')

fig.tight_layout()

plt.show()
