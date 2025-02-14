import numpy as np
import random
from numba import jit, njit

#########################
## Binary Random Grids ##
#########################

def create_random_binary_grid(size, percent_high_k):
    """
    Create a random binary grid with specified percentage of high K cells
    
    Args:
        size (int): Size of the square grid
        percent_high_k (float): Percentage of high K cells (0-100)
        
    Returns:
        numpy.ndarray: Binary grid with high K (1) and low K (0) cells
    """
    total_cells = size * size
    num_high_k = int(total_cells * percent_high_k / 100)
    
    # Create array with required number of 1s and 0s
    grid = np.zeros(total_cells)
    grid[:num_high_k] = 1
    
    # Randomly shuffle the array and reshape to square
    np.random.shuffle(grid)
    return grid.reshape((size, size))


def assign_k_values(binary_grid, k_high, k_low):
    """
    Convert binary grid to conductivity values
    
    Args:
        binary_grid (numpy.ndarray): Binary grid of 0s and 1s
        k_high (float): High conductivity value
        k_low (float): Low conductivity value
        
    Returns:
        numpy.ndarray: Grid with actual K values
    """
    k_grid = np.where(binary_grid == 1, k_high, k_low)
    return k_grid


############################
## Gaussian Random Fields ##
############################

def create_grid(nx, ny, dx=1.0, dy=1.0):
    """Create a 2D grid of points."""
    x = np.linspace(0, (nx-1)*dx, nx)
    y = np.linspace(0, (ny-1)*dy, ny)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def exponential_covariance(h, variance=1.0, correlation_length=1.0):
    """Exponential covariance function."""
    return variance * np.exp(-h/correlation_length)


def gaussian_covariance(h, variance=1.0, correlation_length=1.0):
    """Gaussian covariance function."""
    return variance * np.exp(-(h/correlation_length)**2)


def matern_covariance(h, variance=1.0, correlation_length=1.0, nu=1.5):
    """Matérn covariance function."""
    from scipy.special import gamma, kv
    if nu == 0.5:
        return exponential_covariance(h, variance, correlation_length)
    
    h = np.maximum(h, 1e-10)  # Avoid zero division
    scale = correlation_length
    
    sqrt_2nu = np.sqrt(2 * nu)
    h_scale = sqrt_2nu * h / scale
    
    return variance * (2**(1-nu)) / gamma(nu) * (h_scale**nu) * kv(nu, h_scale)


def generate_grf(nx, ny, covariance_function, variance=1.0, correlation_length=1.0, 
                 dx=1.0, dy=1.0, mean=0.0, nu=1.5):
    """
    Generate a Gaussian Random Field.
    
    Parameters:
    -----------
    nx, ny : int
        Number of points in x and y directions
    covariance_function : function
        Covariance function to use
    variance : float
        Variance of the field
    correlation_length : float
        Correlation length of the field
    dx, dy : float
        Grid spacing in x and y directions
    mean : float
        Mean of the field
    nu : float
        Smoothness parameter for Matérn covariance
    
    Returns:
    --------
    numpy.ndarray
        Generated random field
    """
    # Create grid
    xx, yy = create_grid(nx, ny, dx, dy)
    points = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Calculate distances between all points
    distances = spatial.distance.cdist(points, points)
    
    # Calculate covariance matrix
    if covariance_function.__name__ == 'matern_covariance':
        cov_matrix = covariance_function(distances, variance, correlation_length, nu)
    else:
        cov_matrix = covariance_function(distances, variance, correlation_length)
    
    # Ensure matrix is symmetric and positive definite
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    cov_matrix += 1e-10 * np.eye(len(cov_matrix))
    
    # Generate random field
    L = np.linalg.cholesky(cov_matrix)
    z = np.random.normal(0, 1, nx*ny)
    field = mean + np.dot(L, z)
    
    return field.reshape(ny, nx)


def generate_gaussian_random_binary_field(grf, threshold=0.0):
    """Convert Gaussian random field to binary field."""
    return (grf > threshold).astype(int)


###############################
## Ising Model Random Fields ##
###############################

@njit()
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


def generate_ising_binary_field(N, T, J=1.0, steps=10000):
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

    return ((lattice + 1) // 2)