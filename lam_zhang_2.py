import numpy as np
import matplotlib.pyplot as plt


def simulate_network(load_factors, num_servers, num_simulations):
    """
    Simulates a computer communication network modeled as M/M/1 queues.

    Parameters:
        load_factors (list of float): List of load factors (lambda / mu).
        num_servers (int): Number of servers in the network.
        num_simulations (int): Number of simulations per load factor.

    Returns:
        dict: Dictionary of load factor to simulation results.
    """
    simulation_results = {}

    for load_factor in load_factors:
        if load_factor >= 1.0:
            raise ValueError("Load factor must be less than 1 for system stability.")

        total_delays = []
        for _ in range(num_simulations):
            total_delay = 0
            for _ in range(num_servers):
                queue_delay = np.random.geometric(1 - load_factor) - 1
                total_delay += queue_delay
            total_delays.append(total_delay)

        simulation_results[load_factor] = np.array(total_delays)

    return simulation_results


def main():
    # Parameters
    load_factors = np.linspace(0.3, 0.9, 7)  
    num_servers = 5  
    num_simulations = 50 

    simulation_results = simulate_network(load_factors, num_servers, num_simulations)

    for load_factor, delays in simulation_results.items():
        print(f"Load Factor: {load_factor:.2f}, Mean Delay: {np.mean(delays):.2f}, Std Dev: {np.std(delays):.2f}")

    plt.boxplot([simulation_results[lf] for lf in load_factors], labels=[f"{lf:.2f}" for lf in load_factors])
    plt.xlabel("Load Factor (lambda/mu)")
    plt.ylabel("Total Delay (across servers)")
    plt.title("Computer Communication Network Simulation")
    plt.show()


if __name__ == "__main__":
    main()