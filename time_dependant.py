#Solves the time dependent Schrodinger equation for system with a given Hamiltonian and initial state. 
#It calculates the time evolution of the state using Trotter -Suzuki decomposition

### LIBRARIES ###

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from time_independant import give_plot


### CONSTANTS ###

hbar = constants.hbar

### INPUTS ###

ohm = 2 * np.pi * 10**6         #rabi frequency

delta_max = 2 * np.pi * 50e6

t_max = 1e-6

def delta(t):

    return 8 * delta_max * ( (t-t_max/2) / t_max )**3

phase = 0    

voltages = 2 * np.pi * np.linspace(0, 100e6, 500)      

time = np.linspace(0, 1e-6, 100) #time array
#delta t is contained within this

dt = 1e-6 / 100

psi_0 = np.array([1, 0, 0, 0], dtype=complex) #initial state

def main():

    P_0 = {} #dict to store probabilities for each voltage
    P_b = {} #dict to store probabilities for each voltage for bell

    for v in voltages:

        p_0 = [] #list to store probabilities for each time step
        p_b = [] #list to store probabilities for each time step for bell

        for t in time:

            R, R_dagger, E1, E2, E3, E4 = eigen_t(v, t) #get eigenvectors and eigenvalues

            D = np.array([  [np.exp(-1j*E1*dt), 0, 0, 0],
                            [0, np.exp(-1j*E2*dt), 0, 0],
                            [0, 0, np.exp(-1j*E3*dt), 0],
                            [0, 0, 0, np.exp(-1j*E4*dt)]    ]) #diagonal matrix of time evolution
            
            psi_t = R @ D @ R_dagger @ psi_0 #time evolution of the state

            c1 = psi_t[0] #coefficient of state 1

            bell = 1/np.sqrt(2) * (np.array([0, 1, 1, 0], dtype=complex)) #bell state

            p_0.append(np.abs(c1)**2) #append probability to list
            p_b.append(np.abs(np.dot(bell, psi_t))**2) #append probability to list for bell state

        P_0[v] = p_0 #append list to dictionary
        P_b[v] = p_b #append list to dictionary for bell state

    #Now we have a dictionary with keys as voltages and values as lists of probabilities. 
    #We can now plot the probabilities for each voltage.

    give_plot(P_0, P_b)


def eigen_t(V,t):

    H = hbar/2 * np.array([ [2*delta(t), ohm*np.exp(-1j*phase), ohm*np.exp(-1j*phase), 0], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [0, ohm*np.exp(1j*phase), ohm*np.exp(1j*phase), -2*delta(t)+2*V]]) #Hamiltonian

    eigenvalues, eigenvectors = np.linalg.eigh(H) #eigenvalues and eigenvectors of the Hamiltonian

    E1, E2, E3, E4 = eigenvalues #assigning eigenvalues to variables

    R = eigenvectors #eigenvector matrix

    R_dagger = np.conjugate(R.T) #conjugate transpose of the eigenvector matrix

    return R, R_dagger, E1, E2, E3, E4
