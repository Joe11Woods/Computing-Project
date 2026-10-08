#Solves the time independent Schrodinger equation for system with a given Hamiltonian and initial state. It calculates the time evolution of the state and plots the probability of being in the first state as a function of time.

### LIBRARIES ###

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


### CONSTANTS ###

hbar = constants.hbar

### INPUTS ###

ohm = 2 * np.pi * 10**6         #rabi frequency

delta = 10**6                   #detuning

phase = 0    

voltages = 2 * np.pi * np.linspace(0, 100e6, 1000)      

time = np.linspace(0, 1e-6, 100) #time array

psi_0 = np.array([1, 0, 0, 0], dtype=complex) #initial state


def main():

    P_0 = {} #dict to store probabilities for each voltage
    P_b = {} #dict to store probabilities for each voltage for bell

    for v in voltages:

        p_0 = [] #list to store probabilities for each time step
        p_b = [] #list to store probabilities for each time step for bell

        R, R_dagger, E1, E2, E3, E4 = eigen(v) #get eigenvectors and eigenvalues

        for t in time:

            D = np.array([  [np.exp(-1j*E1*t/hbar), 0, 0, 0],
                            [0, np.exp(-1j*E2*t/hbar), 0, 0],
                            [0, 0, np.exp(-1j*E3*t/hbar), 0],
                            [0, 0, 0, np.exp(-1j*E4*t/hbar)]    ]) #diagonal matrix of time evolution
            
            psi_t = np.dot(R, np.dot(D, np.dot(R_dagger, psi_0))) #time evolution of the state

            c1 = psi_t[0] #coefficient of state 1

            bell = 1/np.sqrt(2) * (np.array([0, 1, 1, 0], dtype=complex)) #bell state

            p_0.append(np.abs(c1)**2) #append probability to list
            p_b.append(np.abs(np.dot(bell, psi_t))**2) #append probability to list for bell state

        P_0[v] = p_0 #append list to dictionary
        P_b[v] = p_b #append list to dictionary for bell state

    #Now we have a dictionary with keys as voltages and values as lists of probabilities. 
    #We can now plot the probabilities for each voltage.

    give_plot(P_0, P_b)


def eigen(V):

    H = hbar/2 * np.array([ [2*delta, ohm*np.exp(-1j*phase), ohm*np.exp(-1j*phase), 0], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [0, ohm*np.exp(1j*phase), ohm*np.exp(1j*phase), -2*delta+2*V]]) #Hamiltonian

    eigenvalues, eigenvectors = np.linalg.eigh(H) #eigenvalues and eigenvectors of the Hamiltonian

    E1, E2, E3, E4 = eigenvalues #assigning eigenvalues to variables

    R = eigenvectors #eigenvector matrix

    R_dagger = np.conjugate(R.T) #conjugate transpose of the eigenvector matrix

    return R, R_dagger, E1, E2, E3, E4

def give_plot(P_0, P_b):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5)) #create plot

    for v in P_0.keys():

        if v == 0:
            ax1.plot(time, P_0[v], color='skyblue') 

        else:
            ax1.plot(time, P_0[v], color='darkred', alpha=0.1)

    for v in P_b.keys():

        if v == 0:
            ax2.plot(time, P_b[v], color='skyblue') 

        else:
            ax2.plot(time, P_b[v], color='darkred', alpha=0.1)


    ax1.set_ylabel('P |00>')


    ax2.set_xlabel('Time (microseconds)')
    ax2.set_ylabel('P (Bell State)')
   

    ax1.set_ylim(0, 1) #set y limits
    ax1.set_xlim(0, 1e-6) #set x limits

    ax2.set_ylim(0, 1) #set y limits
    ax2.set_xlim(0, 1e-6) #set x limits

    plt.savefig('time_independant.png', dpi=300) #save figure

    plt.show()



if __name__ == "__main__":
    main()