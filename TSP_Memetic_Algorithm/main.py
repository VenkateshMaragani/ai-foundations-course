import time
import numpy as np
from tsp.datasets import load_berlin52, load_eil101, load_gr48, load_hk48
from tsp.experiments import run_ma_experiments
from tsp.plots import save_results

if __name__ == "__main__":
    """
    Main script to run the Memetic Algorithm for the Traveling Salesman Problem (TSP) on benchmark datasets.

    This script performs the following steps:
    1. Sets algorithm parameters including number of runs, population size, number of generations, mutation rate,
    local search probability, elite fraction, and randomness in the greedy crossover.
    2. Loads a benchmark dataset (e.g., Berlin52) and its known optimum.
    3. Runs the Memetic Algorithm multiple times using `run_ma_experiments`, collecting best tours, tour lengths,
    runtimes, and convergence curves.
    4. Prints a summary of results including average tour length, best tour length, average runtime, and error
    compared to the known optimum.
    5. Saves visualizations of the results including the best tour, convergence curves, runtime histogram,
    and summary statistics using `save_results`.

    Parameters (set in the script)
    ------------------------------
    runs : int
        Number of independent algorithm runs.
    pop_size : int
        Size of the population for each generation.
    generations : int
        Number of generations to evolve.
    mutation_rate : float
        Probability of mutation applied to offspring.
    local_search_prob : float
        Probability of applying local search on an offspring.
    elite_fraction : float
        Fraction of the population retained as elite in each generation.
    randomness_in_gx : int
        Parameter controlling randomness in the greedy crossover operator.

    Example
    -------
    >>> python main.py
    Inverse Mutation
    Runs: 30
    Population Size: 150
    Generations Size: 500
    Mutation Rate: 0.15
    Local Search Probability: 0.15
    Elite Fraction: 0.15
    Randomness in Crossover: 1
    ...
    Saved plot results/Berlin52_results_run_.png
    """

    runs=30
    pop_size=150
    generations=500
    mutation_rate=0.2
    local_search_prob=0.2
    elite_fraction=0.2
    randomness_in_gx=2
    
    print("Inverse Mutation")
    print("Runs:", runs)
    print("Population Size:" ,pop_size)
    print("Generations Size:" ,generations)
    print("Mutation Rate :" ,mutation_rate)
    print(f"Local Search Probability : {local_search_prob}")
    print(f"Elite Fraction: {elite_fraction}")
    print(f"Randomness in Crossover: {randomness_in_gx}")
    
    cities, optimum = load_berlin52()

    best_tour, lengths, times, mean_curve, std_curve = run_ma_experiments(
        cities,
        runs=runs,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        local_search_prob=local_search_prob,
        elite_fraction=elite_fraction,
        randomness_in_gx=randomness_in_gx
    )
    
    print("\n=== Summary ===")
    print(f"Runs: {len(lengths)}")
    print(f"Avg Length: {np.mean(lengths):.2f}")
    print(f"Best Length: {np.min(lengths):.2f}")
    print(f"Avg Runtime: {np.mean(times):.3f}s\\n")
    

    if optimum:
        error = (np.min(lengths) - optimum) / optimum * 100
        print(f"Optimum: {optimum}, Error: {error:.2f}%")
    
    save_results(cities, best_tour, lengths, times, mean_curve, std_curve, dataset_name="Berlin52", optimum=optimum)
