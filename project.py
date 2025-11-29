import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import pandas as pd


class LeslieMatrix:
    """
    Leslie Matrix class for age-structured population modeling.
    
    The Leslie matrix L is structured as follows:
    - First row: fertility rates (f0, f1, f2, ..., fn-1)
    - Subdiagonal: survival rates (s0, s1, s2, ..., sn-2)
    - All other elements are zero
    
    Example 3x3 Leslie Matrix:
    [f0  f1  f2]
    [s0   0   0]
    [ 0  s1   0]
    """
    
    def __init__(self, fertility_rates, survival_rates):
        """
        Initialize Leslie matrix with fertility and survival rates.
        
        Parameters:
        -----------
        fertility_rates : list or array
            Fertility rates for each age class
        survival_rates : list or array
            Survival rates between age classes (length = len(fertility_rates) - 1)
        """
        self.fertility_rates = np.array(fertility_rates)
        self.survival_rates = np.array(survival_rates)
        self.n_age_classes = len(fertility_rates)
        
        # Validate input dimensions
        if len(survival_rates) != self.n_age_classes - 1:
            raise ValueError("Survival rates must be one less than fertility rates")
        
        self.matrix = self._construct_matrix()
    
    def _construct_matrix(self):
        """Construct the Leslie matrix from fertility and survival rates."""
        L = np.zeros((self.n_age_classes, self.n_age_classes))
        
        # First row: fertility rates
        L[0, :] = self.fertility_rates
        
        # Subdiagonal: survival rates
        for i in range(self.n_age_classes - 1):
            L[i + 1, i] = self.survival_rates[i]
        
        return L
    
    def project_population(self, initial_population, time_steps):
        """
        Project population forward using matrix multiplication.
        
        Parameters:
        -----------
        initial_population : array
            Initial population vector for each age class
        time_steps : int
            Number of time steps to project
        
        Returns:
        --------
        populations : array
            Population matrix where each column is a time step
        """
        initial_pop = np.array(initial_population)
        if len(initial_pop) != self.n_age_classes:
            raise ValueError("Initial population must match number of age classes")
        
        populations = np.zeros((self.n_age_classes, time_steps + 1))
        populations[:, 0] = initial_pop
        
        # Matrix multiplication for each time step
        for t in range(time_steps):
            populations[:, t + 1] = self.matrix @ populations[:, t]
        
        return populations
    
    def eigenvalue_analysis(self):
        """
        Perform eigenvalue analysis of the Leslie matrix.
        
        Returns:
        --------
        dict : Analysis results containing eigenvalues, eigenvectors, 
               dominant eigenvalue, and population growth rate
        """
        eigenvalues, eigenvectors = eig(self.matrix)
        
        # Find dominant eigenvalue (largest real part)
        dominant_idx = np.argmax(np.real(eigenvalues))
        dominant_eigenvalue = eigenvalues[dominant_idx]
        dominant_eigenvector = np.real(eigenvectors[:, dominant_idx])
        
        # Normalize dominant eigenvector to make it a proper age distribution
        stable_age_dist = dominant_eigenvector / np.sum(dominant_eigenvector)
        
        # Population growth rate
        growth_rate = np.real(dominant_eigenvalue)
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'dominant_eigenvalue': dominant_eigenvalue,
            'dominant_eigenvector': dominant_eigenvector,
            'stable_age_distribution': stable_age_dist,
            'population_growth_rate': growth_rate,
            'net_reproductive_rate': growth_rate,
            'generation_time': self._calculate_generation_time()
        }
    
    def _calculate_generation_time(self):
        """Calculate mean generation time."""
        # Simplified calculation using fertility rates
        ages = np.arange(self.n_age_classes)
        return np.sum(ages * self.fertility_rates) / np.sum(self.fertility_rates)
    
    def sensitivity_analysis(self):
        """
        Perform sensitivity analysis of population growth rate to parameter changes.
        
        Returns:
        --------
        dict : Sensitivity matrices for fertility and survival rates
        """
        analysis = self.eigenvalue_analysis()
        w = analysis['dominant_eigenvector']  # Right eigenvector
        
        # Calculate left eigenvector
        eigenvalues_T, eigenvectors_T = eig(self.matrix.T)
        left_idx = np.argmax(np.real(eigenvalues_T))
        v = np.real(eigenvectors_T[:, left_idx])
        
        # Normalize so that v^T * w = 1
        v = v / np.dot(v, w)
        
        # Sensitivity to fertility rates
        fertility_sensitivity = v[0] * w
        
        # Sensitivity to survival rates
        survival_sensitivity = np.zeros(len(self.survival_rates))
        for i in range(len(self.survival_rates)):
            survival_sensitivity[i] = v[i + 1] * w[i]
        
        return {
            'fertility_sensitivity': fertility_sensitivity,
            'survival_sensitivity': survival_sensitivity,
            'left_eigenvector': v,
            'right_eigenvector': w
        }
    
    def reproductive_value(self):
        """Calculate reproductive value for each age class."""
        analysis = self.eigenvalue_analysis()
        sensitivity = self.sensitivity_analysis()
        
        # Reproductive value is proportional to left eigenvector
        repro_values = sensitivity['left_eigenvector']
        # Normalize so that first age class has value 1
        repro_values = repro_values / repro_values[0]
        
        return repro_values
    
    def plot_projection(self, initial_population, time_steps):
        """Plot population projection over time."""
        populations = self.project_population(initial_population, time_steps)
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual age classes
        plt.subplot(2, 2, 1)
        for i in range(self.n_age_classes):
            plt.plot(populations[i, :], label=f'Age class {i}', marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Population Size')
        plt.title('Population Projection by Age Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot total population
        plt.subplot(2, 2, 2)
        total_pop = np.sum(populations, axis=0)
        plt.plot(total_pop, 'k-', linewidth=2, marker='o')
        plt.xlabel('Time Steps')
        plt.ylabel('Total Population')
        plt.title('Total Population Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot age distribution at final time step
        plt.subplot(2, 2, 3)
        final_dist = populations[:, -1] / np.sum(populations[:, -1])
        age_classes = [f'Age {i}' for i in range(self.n_age_classes)]
        plt.bar(age_classes, final_dist)
        plt.xlabel('Age Class')
        plt.ylabel('Proportion')
        plt.title(f'Age Distribution at Time {time_steps}')
        plt.xticks(rotation=45)
        
        # Plot growth rate over time
        plt.subplot(2, 2, 4)
        growth_rates = total_pop[1:] / total_pop[:-1]
        plt.plot(growth_rates, 'r-', linewidth=2, marker='s')
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time Steps')
        plt.ylabel('Growth Rate')
        plt.title('Population Growth Rate Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary_statistics(self):
        """Generate summary statistics for the Leslie matrix."""
        analysis = self.eigenvalue_analysis()
        sensitivity = self.sensitivity_analysis()
        repro_values = self.reproductive_value()
        
        print("="*60)
        print("LESLIE MATRIX ANALYSIS SUMMARY")
        print("="*60)
        print(f"Number of age classes: {self.n_age_classes}")
        print(f"Fertility rates: {self.fertility_rates}")
        print(f"Survival rates: {self.survival_rates}")
        print()
        
        print("LESLIE MATRIX:")
        print(self.matrix)
        print()
        
        print("EIGENVALUE ANALYSIS:")
        print(f"Dominant eigenvalue (λ₁): {analysis['population_growth_rate']:.4f}")
        print(f"Population growth rate: {(analysis['population_growth_rate'] - 1) * 100:.2f}% per time step")
        
        if analysis['population_growth_rate'] > 1:
            print("Population is GROWING")
        elif analysis['population_growth_rate'] < 1:
            print("Population is DECLINING")
        else:
            print("Population is STABLE")
        
        print(f"Generation time: {analysis['generation_time']:.2f} time units")
        print()
        
        print("STABLE AGE DISTRIBUTION:")
        for i, prop in enumerate(analysis['stable_age_distribution']):
            print(f"Age class {i}: {prop:.4f} ({prop*100:.2f}%)")
        print()
        
        print("REPRODUCTIVE VALUES:")
        for i, value in enumerate(repro_values):
            print(f"Age class {i}: {value:.4f}")
        print()
        
        print("SENSITIVITY ANALYSIS:")
        print("Sensitivity to fertility rates:")
        for i, sens in enumerate(sensitivity['fertility_sensitivity']):
            print(f"  f_{i}: {sens:.4f}")
        print("Sensitivity to survival rates:")
        for i, sens in enumerate(sensitivity['survival_sensitivity']):
            print(f"  s_{i}: {sens:.4f}")
    
    def __str__(self):
        """String representation of the Leslie matrix."""
        return f"Leslie Matrix ({self.n_age_classes}x{self.n_age_classes}):\n{self.matrix}"


def create_example_matrices():
    """Create example Leslie matrices for common scenarios."""
    examples = {}
    
    # Example 1: Human population (simplified 3 age classes)
    examples['human'] = LeslieMatrix(
        fertility_rates=[0.0, 0.5, 0.8],  # No reproduction in first age class
        survival_rates=[0.8, 0.7]         # High survival in early ages
    )
    
    # Example 2: Small mammal population (4 age classes)
    examples['small_mammal'] = LeslieMatrix(
        fertility_rates=[0.0, 1.2, 2.5, 1.8],
        survival_rates=[0.6, 0.4, 0.2]
    )
    
    # Example 3: Bird population (5 age classes)
    examples['bird'] = LeslieMatrix(
        fertility_rates=[0.0, 0.8, 1.5, 1.8, 1.2],
        survival_rates=[0.7, 0.8, 0.6, 0.4]
    )
    
    # Example 4: Fish population (high early mortality)
    examples['fish'] = LeslieMatrix(
        fertility_rates=[0.0, 0.0, 50.0, 100.0],
        survival_rates=[0.1, 0.3, 0.5]
    )
    
    return examples


def compare_populations(matrices_dict, initial_pops_dict, time_steps=20):
    """Compare multiple populations side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, matrix) in enumerate(matrices_dict.items()):
        if idx >= 4:  # Only plot first 4
            break
            
        populations = matrix.project_population(
            initial_pops_dict[name], time_steps
        )
        total_pop = np.sum(populations, axis=0)
        
        axes[idx].plot(total_pop, linewidth=2, marker='o')
        axes[idx].set_title(f'{name.capitalize()} Population')
        axes[idx].set_xlabel('Time Steps')
        axes[idx].set_ylabel('Total Population')
        axes[idx].grid(True, alpha=0.3)
        
        # Add growth rate annotation
        analysis = matrix.eigenvalue_analysis()
        growth_rate = analysis['population_growth_rate']
        axes[idx].text(0.05, 0.95, f'λ = {growth_rate:.3f}', 
                      transform=axes[idx].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demonstration of Leslie matrix functionality
    print("Leslie Matrix Analysis Demonstration")
    print("="*50)
    
    # Create example matrices
    examples = create_example_matrices()
    
    # Analyze human population example
    print("\nHUMAN POPULATION EXAMPLE:")
    human_leslie = examples['human']
    human_leslie.summary_statistics()
    
    # Project human population
    initial_human_pop = [1000, 800, 600]  # Initial population by age class
    human_populations = human_leslie.project_population(initial_human_pop, 20)
    
    print(f"\nProjected population after 20 time steps:")
    for i, pop in enumerate(human_populations[:, -1]):
        print(f"Age class {i}: {pop:.0f}")
    print(f"Total: {np.sum(human_populations[:, -1]):.0f}")
    
    # Demonstrate matrix multiplication explicitly
    print(f"\nMatrix multiplication demonstration:")
    print(f"Initial population: {initial_human_pop}")
    print(f"After 1 step: {human_leslie.matrix @ initial_human_pop}")
    print(f"After 2 steps: {human_leslie.matrix @ (human_leslie.matrix @ initial_human_pop)}")
    
    # Plot projections for human population
    human_leslie.plot_projection(initial_human_pop, 20)
    
    # Compare different population types
    initial_populations = {
        'human': [1000, 800, 600],
        'small_mammal': [500, 300, 200, 100],
        'bird': [200, 150, 100, 80, 50],
        'fish': [10000, 1000, 100, 50]
    }
    
    compare_populations(examples, initial_populations)