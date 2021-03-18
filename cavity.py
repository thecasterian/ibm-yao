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
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

uc = u[Nx//2, :]
vc = v[:, Ny//2]

yg = np.array([0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1])
ug = np.array([0, -0.18109, -0.20196, -0.2222, -0.2973, -0.38289, -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1])

xg = np.array([0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1])
vg = np.array([0, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.5155, -0.39188, -0.33714, -0.27669, -0.21388, 0])

ax.plot(2*xc[1:-1]-1, vc[1:-1]/0.6)
ax.plot(uc[1:-1], 2*yc[1:-1]-1)
ax.plot(ug, 2*yg-1, 'ro', markersize=4)
ax.plot(2*xg-1, vg/0.6, 'ro', markersize=4)
ax.legend(['$v_{CL}$', '$u_{CL}$', 'U. Ghia, et. al.'])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal', 'box')
ax.grid(True)

plt.show()
