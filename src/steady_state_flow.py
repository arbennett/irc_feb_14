import numpy as np
import matplotlib.pyplot as plt

from numba import njit

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

@njit()
def _iterate(head, k_grid, convergence_criterion, max_iterations):

    size = k_grid.shape[0]
    iteration = 0
    max_change = 1e99

    while max_change > convergence_criterion and iteration < (max_iterations-1):
        head_old = head.copy()
        
        for i in range(1, size-1):
            for j in range(1, size-1):
                # Harmonic mean of K values
                k_right = 2 * k_grid[i,j] * k_grid[i,j+1] / (k_grid[i,j] + k_grid[i,j+1])
                k_left = 2 * k_grid[i,j] * k_grid[i,j-1] / (k_grid[i,j] + k_grid[i,j-1])
                k_up = 2 * k_grid[i,j] * k_grid[i-1,j] / (k_grid[i,j] + k_grid[i-1,j])
                k_down = 2 * k_grid[i,j] * k_grid[i+1,j] / (k_grid[i,j] + k_grid[i+1,j])
                
                # Update head value
                head[i,j] = (k_right*head[i,j+1] + k_left*head[i,j-1] + 
                           k_up*head[i-1,j] + k_down*head[i+1,j]) / (k_right + k_left + k_up + k_down)
        
        max_change = np.max(np.abs(head - head_old))
        iteration += 1

    return head
 


def solve_flow_problem(k_grid, left_head, right_head, convergence_criterion=0.01):
    """
    Solve steady-state flow problem using finite differences
    
    Args:
        k_grid (numpy.ndarray): Grid of hydraulic conductivity values
        left_head (float): Constant head at left boundary
        right_head (float): Constant head at right boundary
        convergence_criterion (float): Convergence criterion for head values
        
    Returns:
        tuple: (head distribution, flow rates at left and right boundaries)
    """
    size = k_grid.shape[0]
    
    # Initialize head array with linear interpolation between boundaries
    head = np.zeros((size, size))
    for i in range(size):
        head[i, :] = np.linspace(left_head, right_head, size)
    
    max_iterations = 10000
    head = _iterate(head, k_grid, convergence_criterion, max_iterations)
   
    # Calculate flow rates at boundaries
    flow_left = 0
    flow_right = 0
    
    for i in range(size):
        # Left boundary flow
        k_interface = 2 * k_grid[i,0] * k_grid[i,1] / (k_grid[i,0] + k_grid[i,1])
        flow_left += k_interface * (head[i,1] - head[i,0])
        
        # Right boundary flow
        k_interface = 2 * k_grid[i,-1] * k_grid[i,-2] / (k_grid[i,-1] + k_grid[i,-2])
        flow_right += k_interface * (head[i,-1] - head[i,-2])
    
    return head, flow_left, -flow_right  # Negative for right flow due to gradient direction


def calculate_keff(flow_rate, domain_length, domain_height, head_difference):
    """
    Calculate effective hydraulic conductivity using Darcy's Law
    
    Args:
        flow_rate (float): Average flow rate through the system
        domain_length (float): Length of the domain
        domain_height (float): Height of the domain
        head_difference (float): Head difference across the domain
        
    Returns:
        float: Effective hydraulic conductivity
    """
    return flow_rate * domain_length / (domain_height * head_difference)


def calculate_energy_dissipation(head):
    """
    Calculate energy dissipation distribution using equation 3
    E = [∇∅(x, y)]²
    
    Args:
        head (numpy.ndarray): Head distribution from flow solution
        
    Returns:
        numpy.ndarray: Energy dissipation at each point
    """
    # Calculate gradients using central differences
    dy, dx = np.gradient(head, 1.0)  # 1.0 is grid spacing
    
    # Energy dissipation is square of gradient magnitude
    energy = dx*dx + dy*dy
    
    return energy


def calculate_weights(head, head_homogeneous):
    """
    Calculate weighting factors using equation 4
    w(x,y) = [∇∅(x, y)]² / ∫∫[∇∅₀(x, y)]²dxdy
    
    Args:
        head (numpy.ndarray): Head distribution from heterogeneous case
        head_homogeneous (numpy.ndarray): Head distribution from homogeneous case
        
    Returns:
        numpy.ndarray: Weighting factors at each point
    """
    # Calculate energy dissipation for both cases
    energy = calculate_energy_dissipation(head)
    energy_homogeneous = calculate_energy_dissipation(head_homogeneous)
    
    # Calculate weights by normalizing by total energy in homogeneous case
    weights = energy / np.sum(energy_homogeneous)
    
    return weights


def calculate_keff(flow_rate, domain_size=25):
    """
    Calculate effective hydraulic conductivity using Darcy's Law
    Keff = (Q/A) × (dL/dH)
    
    Args:
        flow_rate (float): Flow rate through the system (Q)
        domain_size (int): Size of the domain (number of cells in each direction)
        
    Returns:
        float: Effective hydraulic conductivity
    """
    # A = height of domain (each cell is 1m × 1m)
    cross_section_area = domain_size  # height = domain_size * 1m
    
    # dL = length of domain = domain_size * 1m
    # dH = head difference = 1m (from 2m to 1m)
    # dL/dH = domain_size/1 = domain_size
    hydraulic_gradient = domain_size / 1.0
    
    # Keff = (Q/A) × (dL/dH)
    return (flow_rate / cross_section_area) * hydraulic_gradient


def run_simulation(size=15, percent_high_k=50, k_high=1.0, k_low=0.001, 
                  left_head=2.0, right_head=1.0, binary_grid=None):
    """
    Run complete simulation for a given set of parameters
    
    Args:
        size (int): Grid size
        percent_high_k (float): Percentage of high K cells
        k_high (float): High conductivity value
        k_low (float): Low conductivity value
        left_head (float): Left boundary head
        right_head (float): Right boundary head
        
    Returns:
        tuple: (K_effective, head distribution, K distribution)
    """
    # Create binary grid and assign K values
    if binary_grid is None:
        binary_grid = create_random_binary_grid(size, percent_high_k)
    k_grid = assign_k_values(binary_grid, k_high, k_low)
    
    # Solve flow problem
    head, flow_left, flow_right = solve_flow_problem(k_grid, left_head, right_head)
    
    # Use average of left and right flows
    avg_flow = (abs(flow_left) + abs(flow_right)) / 2
    
    # Calculate effective K
    k_eff = calculate_keff(avg_flow, domain_size=size)
    
    return k_eff, head, k_grid