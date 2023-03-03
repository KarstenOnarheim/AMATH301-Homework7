import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate
import time


##################### Coding Problem 1 ##########################
## Part a - Solve with Adam's Moulton method
# Define the ODE and root-finding problem

## Part b - Implement Adam's Moulton
# Then define tspan and the initial condition.

# Then solve with a for loop!


######################### Coding problem 2 ###################
## Part (a) - setup ODE
# To solve you are going to need the ODEs for y1, y2, and y3. 
# First define the constants
s = 77.27
w = 0.161
q = 1
y1_prime = lambda y1, y2, y3: s*(y2 - y1*y2 + y1 - q*y1**2)
# Now you define y2_prime and y3_prime

# odefun = .... # You define the ode function!

## (b) Solve for 10 logarithmically spaced points, using RK45
A4 = np.zeros([3, 10])

## (c)  Solve for 10 logarithmically spaced points, using BDF
A5 = np.zeros([3, 10])


##################### Coding Problem 3 ##########################
# Define the parameters we are going to use:
mu = 200

## Part a - define the ODEs

## Part b
# Solve using BDF. What are the initial conditions?
x0 = 2
# y0 = ?

## Part c

## Part d

## Part e - linearized version

## Part f - linear system
# Example:
B = np.array([[1, 2], [3, 4]])

## Part g
dt = 0.01

## Part h
# If A is defined, this looks like
#       x[k] + dt*A@x[k]

## Part i
I = np.eye(2)
# To create C, we can just do subtraction and multiplication

