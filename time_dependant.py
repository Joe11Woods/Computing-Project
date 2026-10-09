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
    d = [] #list to store detuning for each time step

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

    for t in time:
        d.append(delta(t) / 2*np.pi) #append detuning to list

    print("Plotting...")
    give_plot_t(P_0, P_b,d, "time_dependant.png")


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


def give_plot_t(P_0, P_b, d, file_name):

    # Convert time from seconds to microseconds
    t_us = time * 1e6

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True)

    # Plot detuning
    ax1.plot(t_us, d, color='darkgreen', linewidth=2)
    ax1.set_ylabel(r'$\Delta(t)$')
    
    # Plot probability of |00>
    for v in P_0:
        if v == 0:
            ax2.plot(
                t_us, P_0[v],
                color='deepskyblue', linewidth=2.5,
                label='V = 0'
            )
        else:
            ax2.plot(
                t_us, P_0[v],
                color='darkred', alpha=0.15, linewidth=1
            )

    # Plot probability of Bell state
    for v in P_b:
        if v == 0:
            ax3.plot(
                t_us, P_b[v],
                color='deepskyblue', linewidth=2.5,
                label='V = 0'
            )
        else:
            ax3.plot(
                t_us, P_b[v],
                color='darkred', alpha=0.15, linewidth=1
            )

    # Labels and titles
    ax2.set_ylabel(r'$P(|00\rangle)$')

    ax3.set_ylabel(r'$P(|\Phi^+\rangle)$')
    ax3.set_xlabel('Time (µs)')

    # Format probability axes
    for ax in (ax2, ax3):
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    
    # Format all axes
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(t_us[0], t_us[-1])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # Final layout and save
    fig.tight_layout()
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    main()
