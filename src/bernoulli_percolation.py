import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ipywidgets import interact, FloatSlider, IntSlider
import ipywidgets as widgets
from numba import jit, njit
import time

# Set default parameters
GRID_SIZE = 30
SAMPLES = 50

@njit
def generate_grid(p, size):
    """Generate Bernoulli percolation grid"""
    return np.random.binomial(1, p, size=(size, size))

@njit
def find_clusters(grid):
    """Find connected clusters using BFS"""
    size = grid.shape[0]
    visited = np.zeros_like(grid, dtype=np.bool_)
    cluster_labels = np.zeros_like(grid, dtype=np.int32)
    current_label = 1
    
    # Pre-allocate queue arrays (max size would be grid size)
    queue_x = np.zeros(size * size, dtype=np.int32)
    queue_y = np.zeros(size * size, dtype=np.int32)
    
    directions = np.array([(-1,0), (1,0), (0,-1), (0,1)])
    
    for i in range(size):
        for j in range(size):
            if grid[i,j] and not visited[i,j]:
                # Initialize queue for new cluster
                queue_start = 0
                queue_end = 1
                queue_x[0] = i
                queue_y[0] = j
                visited[i,j] = True
                cluster_labels[i,j] = current_label
                
                # Process queue
                while queue_start < queue_end:
                    x = queue_x[queue_start]
                    y = queue_y[queue_start]
                    queue_start += 1
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if grid[nx,ny] and not visited[nx,ny]:
                                visited[nx,ny] = True
                                cluster_labels[nx,ny] = current_label
                                queue_x[queue_end] = nx
                                queue_y[queue_end] = ny
                                queue_end += 1
                
                current_label += 1
    
    return cluster_labels, current_label - 1

@njit
def check_percolation(cluster_labels, num_clusters):
    """Check if any cluster percolates"""
    size = cluster_labels.shape[0]
    
    # For each cluster label
    for label in range(1, num_clusters + 1):
        top = False
        bottom = False
        
        # Check top row
        for j in range(size):
            if cluster_labels[0,j] == label:
                top = True
                break
                
        # If found on top, check bottom
        if top:
            for j in range(size):
                if cluster_labels[size-1,j] == label:
                    bottom = True
                    break
        
        if top and bottom:
            return True
            
    return False

@njit
def calculate_stats(p, size, samples):
    """Calculate percolation probability statistics"""
    percolation_count = 0
    for _ in range(samples):
        grid = generate_grid(p, size)
        cluster_labels, num_clusters = find_clusters(grid)
        if check_percolation(cluster_labels, num_clusters):
            percolation_count += 1
    return percolation_count / samples

def plot_percolation(p=0.5, size=GRID_SIZE, samples=SAMPLES):
    """Interactive plot with percolation visualization"""
    # Generate sample grid
    grid = generate_grid(p, size)
    cluster_labels, num_clusters = find_clusters(grid)
    percolates = check_percolation(cluster_labels, num_clusters)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create custom colormap with black for cluster 0 and random turbo colors for others
    turbo = plt.cm.get_cmap('turbo')
    colors = [np.array([0, 0, 0, 1])]  # Black for cluster 0
    if num_clusters > 0:
        random_colors = [turbo(np.random.random()) for _ in range(num_clusters)]
        colors.extend(random_colors)
    custom_cmap = ListedColormap(colors)
    
    # Plot grid with clusters
    ax1.imshow(cluster_labels, cmap=custom_cmap, interpolation='none')
    ax1.set_title(f'Single Realization (Percolates: {percolates})')
    ax1.axis('off')

    # Calculate and plot statistics
    probabilities = np.linspace(0, 1, 50)
    
    # Time the calculation
    start_time = time.time()
    avg_probs = [calculate_stats(prob, size, samples) for prob in probabilities]
    end_time = time.time()
    
    ax2.plot(probabilities, avg_probs, 'b-', label='Simulation')
    ax2.axvline(x=0.5927, color='r', linestyle='--', label='Theoretical p_c')
    ax2.plot(p, calculate_stats(p, size, samples), 'ro', markersize=8)
    ax2.set_xlabel('Probability p')
    ax2.set_ylabel('Percolation Probability')
    ax2.set_title(f'Average over {samples} samples\nComputation time: {end_time-start_time:.2f}s')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
