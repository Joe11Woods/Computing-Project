#Solves the time independent Schrodinger equation for system with a given Hamiltonian and initial state. It calculates the time evolution of the state and plots the probability of being in the first state as a function of time.

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


#constants

hbar = constants.hbar

#inputs

ohm = 10**7      
delta = 10**7
phase = np.pi/4
V = 1
psi_0 = np.array([1, 0, 0, 0], dtype=complex) #initial state
H = hbar/2 * np.array([ [2*delta, ohm*np.exp(-1j*phase), ohm*np.exp(-1j*phase), 0], 
                        [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                        [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                        [0, ohm*np.exp(1j*phase), ohm*np.exp(1j*phase), -2*delta+2*V]]) #Hamiltonian
time = np.linspace(0, 1e-6, 100) #time array

eigenvalues, eigenvectors = np.linalg.eig(H) #eigenvalues and eigenvectors of the Hamiltonian

E1, E2, E3, E4 = eigenvalues #assigning eigenvalues to variables
E1V, E2V, E3V, E4V = eigenvectors.T #assigning eigenvectors to variables

R = np.array([  [E1V[0], E2V[0], E3V[0], E4V[0]],
                [E1V[1], E2V[1], E3V[1], E4V[1]],
                [E1V[2], E2V[2], E3V[2], E4V[2]],
                [E1V[3], E2V[3], E3V[3], E4V[3]]]) #matrix of eigenvectors

R_dagger = np.conjugate(R.T) #conjugate transpose of the eigenvector matrix


p_c1 = [] #list to store c1 probabilities
p_c2 = [] #list to store c2 probabilities
p_c3 = [] #list to store c3 probabilities
p_c4 = [] #list to store c4 probabilities

for t in time:

    D = np.array([[np.exp(-1j*E1*t/hbar), 0, 0, 0],
                  [0, np.exp(-1j*E2*t/hbar), 0, 0],
                  [0, 0, np.exp(-1j*E3*t/hbar), 0],
                  [0, 0, 0, np.exp(-1j*E4*t/hbar)]]) #diagonal matrix of time evolution
    
    psi_t = np.dot(R, np.dot(D, np.dot(R_dagger, psi_0))) #time evolution of the state

    c1 = psi_t[0] #coefficient of state 1
    c2 = psi_t[1] #coefficient of state 2
    c3 = psi_t[2] #coefficient of state 3
    c4 = psi_t[3] #coefficient of state 4

    p_c1.append(np.abs(c1)**2) #append probabilities to list
    p_c2.append(np.abs(c2)**2)
    p_c3.append(np.abs(c3)**2)
    p_c4.append(np.abs(c4)**2)


plot, ax = plt.subplots() #create plot
ax.plot(time, p_c1) #plot c1 probabilities
plt.show()

