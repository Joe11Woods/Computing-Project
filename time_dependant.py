#Solves the time dependent Schrodinger equation for system with a given Hamiltonian and initial state. 
#It calculates the time evolution of the state using Trotter -Suzuki decomposition

### LIBRARIES ###

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


### CONSTANTS ###

hbar = constants.hbar

### INPUTS ###

ohm = 2 * np.pi * 10**6         #RABI FREQUENCY

delta_max = 2 * np.pi * 50e6    #MAXIMUM TUNING

phase = 0    

voltages = 2 * np.pi * np.linspace(0, 100e6, 500) #LIST OF VOLTAGES     

t_max = 1e-6 #MAXIMUM TIME
dt = 1e-8 #TIME STEP
steps = int(t_max / dt)
time = np.arange(steps) * dt

def delta(t): #TIME DEPENDENT DETUNING FUNCTION
    return 8 * delta_max * ( (t-t_max/2) / t_max )**3

psi_0 = np.array([1, 0, 0, 0], dtype=complex) #INITIAL STATE


def main():

    P_0 = {} #dict to store probabilities for each voltage
    P_b = {} #dict to store probabilities for each voltage for bell

    for v in voltages:

        p_0 = [] #list to store probabilities for each time step
        p_b = [] #list to store probabilities for each time step for bell
        states = [psi_0] #list to store states for each time step

        for t in time:

            R, R_dagger, E1, E2, E3, E4 = eigen_t(v, t) #get eigenvectors and eigenvalues

            D = np.array([  [np.exp(-1j*E1*dt/hbar), 0, 0, 0],
                            [0, np.exp(-1j*E2*dt/hbar), 0, 0],
                            [0, 0, np.exp(-1j*E3*dt/hbar), 0],
                            [0, 0, 0, np.exp(-1j*E4*dt/hbar)]    ]) #diagonal matrix of time evolution
            
            psi_t = R @ D @ R_dagger @ states[-1]     #time evolution of the state

            c1 = psi_t[0] #coefficient of state 1
            c1_prob = np.abs(c1)**2 #probability of state 1

            bell = 1/np.sqrt(2) * (np.array([0, 1, 1, 0], dtype=complex)) #bell state
            bell_prob = np.abs(np.vdot(bell, psi_t))**2 #probability of bell state

            states.append(psi_t) #append state to list
            p_0.append(c1_prob) #append probability to list
            p_b.append(bell_prob) #append probability to list for bell state

        P_0[v] = p_0 #append list to dictionary
        P_b[v] = p_b #append list to dictionary for bell state

    #Now we have a dictionary with keys as voltages and values as lists of probabilities. 
    #We can now plot the probabilities for each voltage.

    print("Plotting...")
    give_plot_t(P_0, P_b, "time_dependant.png")


def eigen_t(V,t):

    #Two Quibit Hamiltonian with time dependent detuning
    H = hbar/2 * np.array([ [2*delta(t), ohm*np.exp(-1j*phase), ohm*np.exp(-1j*phase), 0], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [ohm*np.exp(1j*phase), 0, 0, ohm*np.exp(-1j*phase)], 
                            [0, ohm*np.exp(1j*phase), ohm*np.exp(1j*phase), -2*delta(t)+2*V]]) 

    eigenvalues, eigenvectors = np.linalg.eigh(H) #eigenvalues and eigenvectors of the Hamiltonian

    E1, E2, E3, E4 = eigenvalues #assigning eigenvalues to variables

    R = eigenvectors #eigenvector matrix

    R_dagger = np.conjugate(R.T) #conjugate transpose of the eigenvector matrix

    return R, R_dagger, E1, E2, E3, E4

def give_plot_t(P_0, P_b, file_name):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5)) #create plot

    for v in P_0.keys(): #makes v=0 blue

        if v == 0:
            ax1.plot(time, P_0[v], color='skyblue') 

        else:
            ax1.plot(time, P_0[v], color='darkred', alpha=0.1)

    for v in P_b.keys(): #makes v=0 blue

        if v == 0:
            ax2.plot(time, P_b[v], color='skyblue') 

        else:
            ax2.plot(time, P_b[v], color='darkred', alpha=0.1)

        ax1.set_ylabel('P |00>')


    ax2.set_xlabel('Time (microseconds)')
    ax2.set_ylabel('P (Bell State)')
   

    ax1.set_ylim(0, 1) #set y limits
    ax1.set_xlim(0, 1e-6) #set x limits
    ax1.set_xticklabels = False

    ax2.set_ylim(0, 1) #set y limits
    ax2.set_xlim(0, 1e-6) #set x limits

    plt.savefig(file_name, dpi=300) #save figure

    plt.show()

if __name__ == "__main__":
    main()
