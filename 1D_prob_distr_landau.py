# -*- coding: utf-8 -*-
"""
Author: Malen Etxeberria Etxaniz (GitHub: malenetxeberria)

Last update: 10/06/2020

Description: Given Monte Carlo results of the order parameter, this program 
             calculates the probability distribution of the order parameter. 
             It also calculates the corresponding Landau's free energy and 
             fits that curve based on the following equation:
                 
                        F = F0 + a2*Q^2 + a4*Q^4
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
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

filename = "histogram_T36.txt"
savefilep = "file_p36.txt"
savelandau = "landau36.txt"
T = 3.6  # Temperature in which the measurement was made

with open(filename) as f:
    lines = f.readlines()
x = [float(line.split()[0]) for line in lines]

# Symmetrize 
x = [-elem for elem in x] + [elem for elem in x]


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


def histogram(x, r):
    """
    Given any list, this function calculates the corresponding histogram.
    
    Args: 
        x (List): item list
        r (int): number of decimal places to which are rounded the values in x 
        
    Returns:
        xsort (List): sorted list containing all the different values in x
        ysort (List): list containing the number of occurrences of every item in
                      xsort in list x
    """
    x = np.around(x, r)
    x = x.tolist()
    xh, yh = [], []
    xsort, ysort = [], []
    l = len(x)
    for i in range(l):
        xh.append(x[i])
        yh.append(count_val(x, x[i])*1.0/l)
    d = {k: v for k, v in zip(xh, yh)}

    # Unzip dictionary
    for key in sorted(d):
        xsort.append(key)
        ysort.append(d[key])
    return xsort, ysort

    
x, y = histogram(x, 2)

# -----------------------------------------------------------------------------
#                             Save & Load
# -----------------------------------------------------------------------------

points = [x, y]
with open(savefilep, "w") as file:
    for x in zip(*points):
        file.write("{0}\t{1}\n".format(*x))

sys.exit()
    
with open(savefilep) as f1:
    lines = f1.readlines()
x = [float(line.split()[0]) for line in lines]
y = [float(line.split()[1]) for line in lines]

# -----------------------------------------------------------------------------
#                       Interpolate & Smooth
# -----------------------------------------------------------------------------

def interpol_smooth(x, y, points, window_size, poly_order):
    """
    Interpolates and smooths the data by using the following methods: "interp1d" 
    and "savgol_filter", both from the Scipy library. 

    Args: 
        x (List): list containing X coordinates of the data set
        y (List): list containing Y coordinates of the data set
        points (int): number of points in which the data is interpolated
        window_size (int): length of the filter window (i.e., the number of 
                           coefficients); this must be positive, odd and bigger 
                           than or equal to argument points
        poly_order (int): the order of the polynomial used to fit the data
        
    Returns:
        xx (Numpy.ndarray): array containing X coordinates of the interpolated 
                            and smoothed data set
        yy_sg (Numpy.ndarray): array containing Y coordinates of the interpolated 
                               and smoothed data set
    """  
    # Interpolate
    xx = np.linspace(min(x), max(x), points)
    itp = interp1d(x, y)
    
    # Smooth
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    return xx, yy_sg


def remove_borders(rmx, rmy, n):
    """
    Removes first and last n points from both of the given arrays.

    Args: 
        rmx (Numpy.ndarray): list containing X coordinates of the data set
        rmy (Numpy.ndarray): list containing Y coordinates of the data set
        n (int): number of points in which the data is interpolated
        window_size (int): number of points that must be removed from each side
        
    Returns:
        rmx (Numpy.ndarray): list containing X coordinates of the reduced data set 
        rmy (Numpy.ndarray): list containing Y coordinates of the reduced data set
    """  
    ind = []
    dni = []
    l = len(rmx)-n
    for i in range(0, n):
        ind.append(i)
        dni.append(l-1-i)
        
    rmx = np.delete(rmx, ind)
    rmx = np.delete(rmx, dni)
    rmy = np.delete(rmy, ind)
    rmy = np.delete(rmy, dni)
    return rmx, rmy


# Interpolate and smooth
xx, yy_sg = interpol_smooth(x, y, 350, 101, 2) 
xx, yy_sg = remove_borders(xx, yy_sg, 0)

# -----------------------------------------------------------------------------
#                        Plot probability distr.
# -----------------------------------------------------------------------------

fig = plt.figure(facecolor='w', figsize=(16, 9))
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(x, y, 'c', alpha=0.8, lw=0.0, marker='o', markersize=4.0)
ax.plot(xx, yy_sg, 'c', alpha=0.5, lw=1.5, label="$P(Q)$")

ax.set_xlabel('Ordena parametroa')
ax.set_ylabel(' Probabilitate-banaketa')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.set_xlim(-0.8,0.8)

ax.grid(b=True, which='major', c='silver', lw=0.5, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)

for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.savefig("prob_dens_T36.png", dpi=600)
plt.show()

# -----------------------------------------------------------------------------
#                          Landau's Free Energy
# -----------------------------------------------------------------------------

# Calculate Landau's free energy (per-site)
ylandau = -T*np.log(y)/1000.0

# Interpolate and smooth
#xx, ylandau_sg = interpol_smooth(x, ylandau, 350, 101, 4)


def extrapolate(x, y, newx, poly_order):
    """
    Extrapolates the data by using the following methods: "polyfit" and "polyval",
    both from the NumPy library. 

    Args: 
        x (Numpy.ndarray): array containing X coordinates of the data set
        y (Numpy.ndarray): array containing Y coordinates of the data set
        newx (Numpy.ndarray): array containing extended X coordinates
        poly_order (int): the order of the polynomial used to fit the data
                          (only possible values: 2 and 4)
                          
    Returns:
        newx (Numpy.ndarray): array containing X coordinates of the extrapolated 
                              data set
        newy (Numpy.ndarray): array containing Y coordinates of the extrapolated 
                              data set
    """  
    if poly_order == 2: 
        a, b, c = np.polyfit(x, y, 2)
        newy  = np.polyval([a,b,c], newx)

    elif poly_order == 4:
        a, b, c, d, e = np.polyfit(x, y, 4)
        newy = np.polyval([a,b,c,d,e], newx)
        
    return newx, newy

# -----------------------------------------------------------------------------
#                                  Fit
# -----------------------------------------------------------------------------

def fit(x, y, order): 
    """
    Given X and Y coordinates corresponding to Landau's free energy, this
    function fits the curve with 4th and 6th degree polynomials. 

    Args: 
        x (List): list containing X coordinates of the data set
        y (List): list containing Y coordinates of the data set
        order (int): the order of the polynomial used to fit the data
                     (only possible values: 4 and 6)

    Returns:
        yfit (List): list containing Y coordinates of the fitted 
                     data set
        coef (Numpy.ndarray): polynomial coefficients, highest power first
        res (float): sum of squared residuals of the least-squares fit
    """  
    m = np.polyfit(x, y, order, full=True)
    coef = m[0]
    res = float(m[1]) 
    
    yfit = []
    
    if order == 4:
        for i in range(0, len(x)):
            yfit.append(coef[0]*x[i]**4 + coef[1]*x[i]**3 + coef[2]*x[i]**2 
                      + coef[3]*x[i] + coef[4])
    elif order == 6:
        for i in range(0, len(x)):
            yfit.append(coef[0]*x[i]**6 + coef[1]*x[i]**5 + coef[2]*x[i]**4 
                      + coef[3]*x[i]**3 + coef[4]*x[i]**2 + coef[5]*x[i] + coef[6])
        
    return yfit, coef, res


# Fit
yfit, coef, res = fit(x, ylandau, 4)
print(coef)

# -----------------------------------------------------------------------------
#                                  R-squared
# -----------------------------------------------------------------------------

def R_squared(y, yest, res):
    """
    This function calculates the coefficient of determination based on its 
    definition.
    
    Args: 
        y (List): list containing Y coordinates of the original data set
        yest (List): list containing Y coordinates of the fitted data set
        res (float): sum of squared residuals of the least-squares fit
            
    Returns:
        multiR2 (float): multiple coefficient of determination
        adjR2 (float): adjusted ccoefficient of determination
    """    
    n = len(y)
    ymean = sum(y)/n
    SStot = 0.0
    for elem in y:
        SStot += (elem-ymean)**2
    
    R2 = 1 - (res/SStot)
    return R2


R2 = R_squared(ylandau, yfit, res)
print(R2)

# -----------------------------------------------------------------------------
#                       Plot Landau's free energy
# -----------------------------------------------------------------------------

fig = plt.figure(facecolor='w', figsize=(16,9))
ax = fig.add_subplot(111, axisbelow=True)

ax.plot(x, ylandau, 'c', alpha=0.75, lw=0.0, marker='o', markersize=4)
#ax.plot(xx, ylandau_sg, 'c', alpha=0.65, lw=1.1, label="$k_B T=3.3$")
ax.plot(x, yfit, "black", alpha=0.65, lw=1.1)

ax.set_xlabel('Ordena parametroa',labelpad=8, fontsize=22.5)
ax.set_ylabel('Landauren energia askea',labelpad=8, fontsize=22.5)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0) 

ax.grid(b=True, which='major', c='silver', lw=0.5, ls='-')
legend = ax.legend(loc=1)
legend.get_frame().set_alpha(0.85)

for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)

plt.savefig("landau_T36.png", dpi=600)
plt.show()