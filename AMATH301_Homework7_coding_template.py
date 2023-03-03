import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import time


##################### Coding Problem 1 ##########################
## Part a - Solve with Adam's Moulton method
# Define the ODE and root-finding problem
odefun = lambda t, y: 5e5 * (-y + np.sin(t))
tspan = np.linspace(0, 2*np.pi, 100)
y0 = 0

def AdamsMoulton(odefun, tspan, y0):
    dt = tspan[1] - tspan[0]
    y = np.zeros(len(tspan))
    y[0] = y0
    for k in range(len(tspan) - 1):
        g = lambda z: z - y[k] - dt*.5*(odefun(tspan[k+1], z) + odefun(tspan[k], y[k]))
        y[k+1] = scipy.optimize.fsolve(g, y[k])
    return [tspan, y]

A1 = 3 - (tspan[1]-tspan[0])*.5*(odefun(tspan[1], 3) + odefun(0, 0))
A2 = AdamsMoulton(odefun, tspan, y0)[1]


######################### Coding problem 2 ###################
## Part (a) - setup ODE
# To solve you are going to need the ODEs for y1, y2, and y3. 
# First define the constants
s = 77.27
w = 0.161
q = 1
y1_prime = lambda y1, y2, y3: s*(y2 - y1*y2 + y1 - q*y1**2)
y2_prime = lambda y1, y2, y3: 1/s * (-y2 - y1*y2 + y3)
y3_prime = lambda y1, y2, y3: w*(y1 - y3)
    
odefun = lambda t, y: [y1_prime(y[0], y[1], y[2]),
                       y2_prime(y[0], y[1], y[2]),
                       y3_prime(y[0], y[1], y[2])]
A3 = odefun(1, [2, 3, 4])
## (b) Solve for 10 logarithmically spaced points, using RK45
y0 = [1, 2, 3]

A4 = np.zeros([3, 10])
qvals = np.logspace(1, 1e-5, 10)
counter = 0
for q in qvals:
    yFinal = scipy.integrate.solve_ivp(odefun, [0,30], y0).y
    A4[0][counter] = yFinal[0][-1]
    A4[1][counter] = yFinal[1][-1]
    A4[2][counter] = yFinal[2][-1]
    counter = counter + 1

## (c)  Solve for 10 logarithmically spaced points, using BDF
A5 = np.zeros([3, 10])
counter = 0
for q in qvals:
    yFinal = scipy.integrate.solve_ivp(odefun, [0,30], y0, method="BDF").y
    A5[0][counter] = yFinal[0][-1]
    A5[1][counter] = yFinal[1][-1]
    A5[2][counter] = yFinal[2][-1]
    counter = counter + 1

##################### Coding Problem 3 ##########################
# Define the parameters we are going to use:
mu = 200

## Part a - define the ODEs
dydt = lambda x, y: mu*(1 - x**2)*y - x
dxdt = lambda x, y: y

A6 = dxdt(2,3)
A7 = dydt(2,3)
## Part b
# Solve using BDF. What are the initial conditions?
x0 = 2
y0 = 0

odefun = lambda t, xy: [dxdt(xy[0], xy[1]), dydt(xy[0], xy[1])]
sol3RK45 = scipy.integrate.solve_ivp(odefun, [0, 400], [x0, y0]).y
A8 = sol3RK45[0]
## Part c
sol3BDF = scipy.integrate.solve_ivp(odefun, [0, 400], [x0, y0], method="BDF").y
A9 = sol3BDF[0]
## Part d
A10 = len(A8)/len(A9)
## Part e - linearized version
dydtLin = lambda x, y: mu*y - x
dxdtLin = lambda x, y: y

A11 = dxdtLin(2,3)
A12 = dydtLin(2,3)
## Part f - linear system
A = np.array([[0, 1],
              [-1, mu]])
A13 = A
## Part g
dt = 0.01
tspan = np.arange(0, 400 + dt, dt)
x = np.zeros([len(tspan), 2])
x[0] = [2,3]
## Part h
#for k in range(len(tspan) - 1):
#    x[k+1] = x[k] + dt*A@x[k]
#A14 = x

## Part i
x = np.zeros([len(tspan), 2])
x[0] = [2,3]
I = np.eye(2)
C = I - dt*A
A15 = C

for k in range(len(tspan) - 1):
    x[k+1] = np.linalg.solve(C, x[k])
xvals = np.zeros(len(x))
for k in range(len(x)):
    xvals[k] = x[k][0]
A16 = xvals
# To create C, we can just do subtraction and multiplication


