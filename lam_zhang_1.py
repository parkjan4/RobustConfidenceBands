import numpy as np
import matplotlib.pyplot as plt

def simulate_mm1(arrival_rate, service_rate, num_simulations):
    """
    Simulates an M/M/1 queue.
    
    Parameters:
        arrival_rate (float): Arrival rate (lambda).
        service_rate (float): Service rate (mu).
        num_simulations (int): Number of simulations to run.
    
    Returns:
        np.ndarray: Array of steady-state numbers of customers.
    """
    if arrival_rate >= service_rate:
        raise ValueError("Arrival rate must be less than service rate to ensure system stability.")
    
    results = []
    for _ in range(num_simulations):
        rho = arrival_rate / service_rate
        num_customers = np.random.geometric(1 - rho) - 1 
        results.append(num_customers)
    
    return np.array(results)

def main():
    service_rate = 1.0  # mu
    # arrival_rates = np.linspace(0.3, 0.9, 7)
    arrival_rates = np.arange(0.3, 0.9, 0.02)
    num_simulations_per_point = 50

    simulation_data = {}
    for rate in arrival_rates:
        data = simulate_mm1(rate, service_rate, num_simulations_per_point)
        simulation_data[rate] = data
    
    for rate, data in simulation_data.items():
        print(f"Arrival Rate: {rate:.2f}, Mean Customers: {np.mean(data):.2f}, Std Dev: {np.std(data):.2f}")

    plt.boxplot([simulation_data[rate] for rate in arrival_rates], labels=[f"{rate:.2f}" for rate in arrival_rates])
    plt.xlabel("Arrival Rate (λ)")
    plt.ylabel("Steady-State Number of Customers")
    plt.title("M/M/1 Queue Simulation")
    plt.show()

if __name__ == "__main__":
    main()
