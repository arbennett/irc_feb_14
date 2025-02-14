import numpy as np
from numba import jit
import random
import matplotlib.pyplot as plt

@jit(nopython=True)
def metropolis_step(lattice, beta, J):
    """Perform one Metropolis step on the lattice."""
    N = len(lattice)
    for _ in range(N*N):
        # Choose random site
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
    
        # Calculate energy change for flip
        spin = lattice[i, j]
        neighbors = (lattice[(i+1)%N, j] + 
                    lattice[i, (j+1)%N] + 
                    lattice[(i-1)%N, j] + 
                    lattice[i, (j-1)%N])
    
        delta_E = 2 * J * spin * neighbors
    
        # Metropolis acceptance criterion
        if delta_E <= 0 or random.random() < np.exp(-beta * delta_E):
            lattice[i, j] = -spin

def generate_ising_grid(N, T, J=1.0, steps=10000):
    """
    Run Ising model simulation.
    
    Parameters:
    -----------
    N : int
        Size of the lattice (N x N)
    T : float
        Temperature in units where k_B = 1
    J : float
        Coupling constant
    steps : int
        Number of Monte Carlo steps
    equilibration_steps : int
        Number of steps to equilibrate before measurements
        
    Returns:
    --------
    dict containing energy, magnetization, and their fluctuations
    """
    # Initialize random lattice
    lattice = np.random.choice([-1, 1], size=(N, N))
    beta = 1.0 / T
    
    # Main simulation loop with measurements
    for _ in range(steps):
        metropolis_step(lattice, beta, J)
   
    return lattice