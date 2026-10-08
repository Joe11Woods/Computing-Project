#Solves the time independent Schrodinger equation for system with a given Hamiltonian and initial state. It calculates the time evolution of the state and plots the probability of being in the first state as a function of time.

### LIBRARIES ###

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


### CONSTANTS ###

hbar = constants.hbar

### INPUTS ###

ohm = 2 * np.pi * 10**6         #rabi frequency

delta = 10**6       #detuning

phase = np.pi/4     

voltages = 2 * np.pi * np.linspace(0, 100e6, 1000)      

time = np.linspace(0, 1e-6, 100) #time array

psi_0 = np.array([1, 0, 0, 0], dtype=complex) #initial state


def eigen(V):

    H = hbar/2 * np.array([ [2*delta, ohm*np.exp(-1j*phase), ohm*np.exp(-1j*phase), 0], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [0, ohm*np.exp(1j*phase), ohm*np.exp(1j*phase), -2*delta+2*V]]) #Hamiltonian

    eigenvalues, eigenvectors = np.linalg.eig(H) #eigenvalues and eigenvectors of the Hamiltonian

    E1, E2, E3, E4 = eigenvalues #assigning eigenvalues to variables

    R = eigenvectors #eigenvector matrix

    R_dagger = np.conjugate(R.T) #conjugate transpose of the eigenvector matrix

    return R, R_dagger, E1, E2, E3, E4

def main():

    P = {} #dict to store probabilities for each voltage

    for v in voltages:

        p = [] #list to store probabilities for each time step

        R, R_dagger, E1, E2, E3, E4 = eigen(v) #get eigenvectors and eigenvalues

        for t in time:

            D = np.array([  [np.exp(-1j*E1*t/hbar), 0, 0, 0],
                            [0, np.exp(-1j*E2*t/hbar), 0, 0],
                            [0, 0, np.exp(-1j*E3*t/hbar), 0],
                            [0, 0, 0, np.exp(-1j*E4*t/hbar)]    ]) #diagonal matrix of time evolution
            
            psi_t = np.dot(R, np.dot(D, np.dot(R_dagger, psi_0))) #time evolution of the state

            c1 = psi_t[0] #coefficient of state 1

            p.append(np.abs(c1)**2) #append probability to list

        P[v] = p #append list to dictionary

    #Now we have a dictionary with keys as voltages and values as lists of probabilities for each time step. 
    #We can now plot the probabilities for each voltage.

    give_plot(P, title='Probability of Being in State 1')


def give_plot(P, title):

    plot, ax = plt.subplots() #create plot

    for v in P.keys():

        if v == 0:
            ax.plot(time, P[v], color='blue') 

        else:
            ax.plot(time, P[v], color='red', alpha=0.2)



    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.set_title(title)

    ax.set_ylim(0, 1) #set y limits
    ax.set_xlim(0, 1e-6) #set x limits



    plt.show()




if __name__ == "__main__":
    main()