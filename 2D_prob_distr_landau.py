# -*- coding: utf-8 -*-
"""
Author: Malen Etxeberria Etxaniz (GitHub: malenetxeberria)

Last update: 16/07/2020

Description: Given Monte Carlo results of both order parameters (which are written
             in a single file), this program calculates the 2D probability distribution
             of the order parameters. It also calculates the corresponding Landau's
             free energy and fits that surface based on the following equation:
                 
                    F = F0 + a2*Q1^2 + a4*Q1^4 + b2*Q2^2 + c*Q1*Q2 
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import sys

# -----------------------------------------------------------------------------
#                                LaTex
# -----------------------------------------------------------------------------

matplotlib.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font = {'weight': 'normal', 'size': 22}  # Graph number's fontsize
plt.rc('font', **font)
plt.rc('legend', fontsize=11)  # Legend's fontsize

# -----------------------------------------------------------------------------
#                 File opening & Histogram calculations
# -----------------------------------------------------------------------------

filename = "bc_hist_T39"
T = 3.9  # Temperature in which the measurement was made

with open(filename+".txt") as f:
    lines = f.readlines()
q1 = [float(line.split()[0]) for line in lines]  # List of measured values of Q1
q2 = [float(line.split()[1]) for line in lines]  # List of measured values of Q2

# Round values
r = 2
for i in range(len(q1)):
    x1, x2 = q1[i], q2[i]
    q1[i] = round(x1, r)
    q2[i] = round(x2, r)
    
# Symmetrize both order parameter arrays
q1 = [-elem for elem in q1] + [elem for elem in q1]
q2 = [-elem for elem in q2] + [elem for elem in q2]


def count_val(x, val):
    """
    Returns the number of occurrences of given item in given list.
    
    Args: 
        x (List): item list
        val (float): item whose occurrences are counted
        
    Returns:
        c (int): number of occurrences
    """
    c = 0  # Counter
    for i in range(len(x)):
        if x[i]==val:
            c += 1
    return c


def histogram(x):
    """
    Given any list, this function calculates the corresponding histogram.
    
    Args: 
        x (List): item list
        
    Returns:
        hist (List): sorted list of two element tuples in which the firts one
                     indicates an item in list x and the second one the number 
                     of its occurrences
    """
    xh, yh = [], []
    l = len(x)
    for i in range(l):
        xh.append(x[i])
        yh.append(count_val(x, x[i])*1.0/l)    
    d = {k: v for k, v in zip(xh, yh)}
    
    # Convert dictionary to tuple list and sort
    hist = sorted(d.items()) 
    return hist


# Calculate Q1's histogram 
p1 = histogram(q1)
q1_vals = [t[0] for t in p1]
p1_vals = [t[1] for t in p1]

# Calculate Q2's histogram
p2 = histogram(q2) 
q2_vals = [t[0] for t in p2]
p2_vals = [t[1] for t in p2]

# -----------------------------------------------------------------------------
#                         Probability Distribution
# -----------------------------------------------------------------------------

def related_q1_q2(q1i, q1, q2):
    """
    Returns a list containing all the items in q2 which are measured along with 
    q1i. This function is necessary so that the general multiplication rule of 
    probabilities can be used.
    
    Args:
        q1i (float): any item in list q1
        q1 (List(1, np)): list of all the measured values of Q1
        q2 (List(1, np)): list of all the measured values of Q2
        
    Returns:
        q2q1 (List): list of element in q2 which are measured along with q1i
    """
    q2q1 = []
    for i in range(len(q1)):
        if q1i == q1[i]:
            q2q1.append(q2[i])
    return q2q1


def prob_2d(p1, q1, q2, q2_vals): 
    """
    Returns de 2D probability distribution of the order parameters.
    
    Args: 
        p1 (List(1, n1)): histogram corresponding to Q1
        q1 (List(1, np)): list of all the measured values of Q1
        q2 (List(1, np)): list of all the measured values of Q2
        q2_vals (List(1, n2)): list of unrepeated measured values of Q2
            
    Returns:
        P (Numpy.array(n1, n2)): two dimensional probability distribution
    """
    rows = len(p1)
    cols = len(q2_vals)
    P = np.zeros((rows, cols))

    for i in range(rows):
        q1i = p1[i][0]
        p1i = p1[i][1]
        q2q1 = related_q1_q2(q1i, q1, q2)

        if not q2q1:
            P[i][0:cols] = 0.0

        else:
            for j in range(cols):
                q2j = q2_vals[j]
                if q2j in q2q1:
                    c = count_val(q2q1, q2j)
                    p2j = c/len(q2q1)
                    P[i][j] = p1i*p2j
                else:
                    P[i][j] = 0.0

    return P


P = prob_2d(p1, q1, q2, q2_vals)
# Transpose for np.meshgrid
P = np.transpose(P)

# Convert List to Numpy.array
q1_vals = np.array(q1_vals)
q2_vals = np.array(q2_vals)

# -----------------------------------------------------------------------------
#                             Save & Load
# -----------------------------------------------------------------------------

np.savetxt('P_T'+str(T)+'.txt', P)
np.savetxt('q1_T'+str(T)+'.txt', q1_vals)
np.savetxt('q2_T'+str(T)+'.txt', q2_vals)

sys.exit()

P = np.loadtxt('P_T'+str(T)+'.txt')
q1_vals = np.loadtxt('q1_T'+str(T)+'.txt')
q2_vals = np.loadtxt('q2_T'+str(T)+'.txt')

# -----------------------------------------------------------------------------
#                         Smooth & Interpolate
# -----------------------------------------------------------------------------

q1mesh, q2mesh = np.meshgrid(q1_vals, q2_vals)  # Mesh of order parameters

points = 200  
q1_vals2 = np.linspace( q1_vals[0], q1_vals[len(q1_vals)-1], points )
q2_vals2 = np.linspace( q2_vals[0], q2_vals[len(q2_vals)-1], points )
q1mesh2, q2mesh2 = np.meshgrid( q1_vals2, q2_vals2 )  # New mesh (more points)

# Smooth surface
tck = interpolate.bisplrep( q1mesh, q2mesh, P, kx=5, ky=5, s=2 )
Psmooth = interpolate.bisplev( q1mesh2[0, :], q2mesh2[:, 0], tck )

# Interpolate P
f = interpolate.interp2d( q1_vals, q2_vals, P, kind='cubic')
Pinterp = f( q1_vals2, q2_vals2 )

# Multiply smoothed and interpolated
Pnew = np.zeros((points, points))
for i in range(points):
    for j in range(points):
        Pnew[i][j] = Psmooth[i][j]*Pinterp[i][j]

# Scale
Pmax_old = np.max(P)
Pmax_new = np.max(Pnew)
Pnew = Pnew*Pmax_old/Pmax_new

# -----------------------------------------------------------------------------
#                        Plot probability distr.
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(25,12))
ax = fig.gca(projection='3d')

# Plot the surface
#surf = ax.plot_surface(q1mesh, q2mesh, P, cmap=cm.coolwarm, rstride=1, cstride=1, antialiased=False)
surf = ax.plot_surface(q1mesh2, q2mesh2, Pnew, cmap=cm.coolwarm, rstride=1, cstride=1, antialiased=False)

ax.set_xlabel('$Q_1$ ordena parametroa',labelpad=45, fontsize=30.5)
ax.set_ylabel('$Q_2$ ordena parametroa',labelpad=49, fontsize=30.5)
ax.set_zlabel("Probabilitate-banaketa",labelpad=23, fontsize=30.5 )

ax.set_zlim(0.000265, 0.01075)
ax.set_zticks([0, 0.002, 0.004, 0.006, 0.008, 0.010])
#ax.set_ylim(-0.065, 0.065)
ax.set_yticks([ -0.08, -0.04, 0.0, 0.04, 0.08])
ax.set_xlim(-0.193, 0.192)
ax.set_xticks([ -0.2, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.2])

plt.gca().patch.set_facecolor('white')
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

plt.rcParams['grid.color'] = "silver"
matplotlib.rc('axes',edgecolor='silver')

#ax.view_init(20, 50)
#ax.view_init(90, 0)

#plt.savefig("prob_dens_T39.png", dpi=600)
plt.show() 

# -----------------------------------------------------------------------------
#                          Landau's Free Energy
# -----------------------------------------------------------------------------

# Lift surface so that there's no log(0)
Pnew = Pnew+1e-4

# Calculate Landau's free energy (per-site)
Fnew = -T*np.log(Pnew)/1000.0  

# Scale
Fmin_teo = -T*np.log(Pmax_old)/1000.0
Fmin = np.min(Fnew)
dif = Fmin_teo - Fmin
Fnew = Fnew + dif

# Make nan the horizontal plane (points corresponing to P = 0.0)
eps = 1e-4
F0 = Fnew[0][0]
F02 = F0-eps
for i in range(points):
    for j in range(points):
        if Fnew[i][j] == F0 or Fnew[i][j] > F02:
            Fnew[i][j] = np.nan

    
def central_sym(Fnew, points):
    """
    Symmetrizes Landau's free energy (central symmetry). Although Fnew already 
    has central symmetry, this step is taken so that the upcoming fit is better.
    
    Args: 
        Fnew (Numpy.array(points, points)): Landau's free energy matrix
        points (int): Fnew's length in each dimension
            
    Returns:
        Fsym (Numpy.array(points, points)): symmetrized Landau's free energy
    """    
    Fsym = Fnew.copy()

    if points%2 == 0:  # Even length
        l = int(points/2.0)
        for i in range(points-1, l-1, -1):
            row = Fnew[2*l - i -1] 
            rowrev = row.copy()
            rowrev = np.fliplr([rowrev])[0]
            Fsym[i] = rowrev
            
    else:  # Odd length
        l = int((points-1)/2.0)
        for i in range(points-1, l, -1):
            row = Fnew[2*l - i]
            rowrev = row.copy()
            rowrev = np.fliplr([rowrev])[0]
            Fsym[i] = rowrev
        for j in range(l):
            elem = Fnew[l][j]
            Fsym[l][points-1-j] = elem

    return np.array(Fsym)

# Symmetrize
Fnew = central_sym(Fnew, points)

# -----------------------------------------------------------------------------
#                                  Fit
# -----------------------------------------------------------------------------

# Flatten in order to remove nans
xf = q1mesh2.flatten()
yf = q2mesh2.flatten()
zf = Fnew.flatten()

points_nan = []
for i in range(len(zf)):
    points_nan.append([xf[i], yf[i], zf[i]])

points_nan = np.array(points_nan)
            
# Remove nans
points_no_nan = []
for elem in points_nan:
    if np.logical_not(np.isnan(elem[2])):
        points_no_nan.append(elem)

points_no_nan = np.array(points_no_nan)
x, y, z = points_no_nan.T


def poly_matrix(x, y):
    """
    Generates the suitable "coefficient" matrix so that "Numpy.linalg.lstsq"
    method can be used. The order of the fit that it is going to be performed
    is chosen here, and it corresponds to the following equation:
        
            F = F0 + a2*Q1^2 + a4*Q1^4 + b2*Q2^2 + c*Q1*Q2 
    
    More information about the method used:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    
    Args: 
        x (Numpy.ndarray(1, points)): q1mesh2 once it has been flattened and nans removed
        y (Nunpy.ndarray(1, points)): q2mesh2 once it has been flattened and nans removed
            
    Returns:
        G (Numpy.array(points, 5)): coefficient matrix
    """    
    G = np.zeros((x.size, 5))
    ij = ((0, 0), (2, 0), (4, 0), (0, 2), (1, 1))  # Q1 and Q2's exponents 
    
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j

    return G


# Calculate coefficient matrix
G = poly_matrix(x, y)

# Solve for np.dot(G, m) = z
m_res = np.linalg.lstsq(G, z)

# Coefficients: F0, a2, a4, b2, c
m = m_res[0]
print(m)

# Sum of squared residuals
res = float(m_res[1])

# -----------------------------------------------------------------------------
#                             Plot fitting solution
# -----------------------------------------------------------------------------

GG = poly_matrix(q1mesh2.ravel(), q2mesh2.ravel())
zz = np.reshape(np.dot(GG, m), q1mesh2.shape)

fig = plt.figure(figsize=(25,12))
ax = fig.gca(projection='3d')

# Plot fitting solution
norm = matplotlib.colors.Normalize(vmin = np.nanmin(Fnew), vmax=0.040, clip = False)
surf = ax.plot_surface(q1mesh2, q2mesh2, zz, cmap=cm.coolwarm, rstride=1, cstride=1, alpha=0.4)
ax.contourf(q1mesh2, q2mesh2, zz, 200, zdir='z', offset=0.0, cmap=cm.coolwarm, norm=norm, vmax=0.04, alpha=0.75)

ax.set_zlim(0.0, 0.040)

# Plot original points
ax.plot3D(x, y, z, "black", marker="o", markersize=2.0, lw=0.0)
plt.show()

# -----------------------------------------------------------------------------
#                             Adjusted R-squared
# -----------------------------------------------------------------------------

def R_squared(Fnew, points, res):
    """
    This function calculates the multiple and adjusted coefficients of determination
    based on their definition.
    
    Args: 
        Fnew (Numpy.array(points, points)): Landau's free energy matrix
        points (int): Fnew's length in each dimension
        res (float): sum of squared residuals
            
    Returns:
        multiR2 (float): multiple coefficient of determination
        adjR2 (float): adjusted ccoefficient of determination
    """    
    n = 0.0
    m = 2.0
    ymean = 0.0
    for i in range(points):
        for j in range(points):
            if not np.isnan(Fnew[i][j]):  # nans are not taken into account
                n += 1
                ymean += Fnew[i][j]
    ymean = ymean/n
    
    SStot = 0.0
    for i in range(points):
        for j in range(points):
            if not np.isnan(Fnew[i][j]):
                SStot += (Fnew[i][j] - ymean)**2
                
    multiR2 = 1 - (res/SStot)
    adjR2 = 1 - (1-multiR2)*(n-1)/(n-m-1)
    
    return multiR2, adjR2


multiR2, adjR2 = R_squared(Fnew, points, res)
print(multiR2, adjR2)

# -----------------------------------------------------------------------------
#                       Plot Landau's free energy
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(25,12))
ax = fig.gca( projection= '3d')

norm = matplotlib.colors.Normalize(vmin = np.nanmin(Fnew), vmax = 0.040, clip = False)
surf = ax.plot_surface(q1mesh2, q2mesh2, Fnew, cmap=cm.coolwarm,  norm=norm, vmax = 0.040, rstride=1, cstride=1,  antialiased=False)
ax.contourf(q1mesh2, q2mesh2, Fnew, 100, zdir='z', offset=0.00, cmap=cm.coolwarm, norm=norm, vmax=0.04, alpha=0.75)

ax.set_xlabel('$Q_1$ ordena parametroa',labelpad=45, fontsize=30.5)
ax.set_ylabel('$Q_2$ ordena parametroa',labelpad=49, fontsize=30.5)
ax.set_zlabel("Landauren energia askea",labelpad=23, fontsize=30.5 )
  
ax.set_zlim(0.00, 0.040)
ax.set_zticks([0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
#ax.set_ylim(-0.065, 0.065)
ax.set_yticks([ -0.08, -0.04, 0.0, 0.04, 0.08])
ax.set_xlim(-0.193, 0.192)
ax.set_xticks([ -0.2, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.2])

plt.gca().patch.set_facecolor('white')
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

plt.rcParams['grid.color'] = "silver"
matplotlib.rc('axes',edgecolor='silver')

#ax.view_init(90, 0)

#plt.savefig("landau_T39.png", dpi=600)
plt.show()