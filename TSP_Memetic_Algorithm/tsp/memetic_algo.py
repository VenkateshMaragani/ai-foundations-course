import random
import numpy as np

from .distance_fun import create_distance_matrix, total_distance
from .greedy_crossover import greedy_crossover_stochastic
from .mutation import mutate_swap
from .hybrid_ls import hybrid_local_search

def roulette_select(population, fitness_vals):
    """
    Select an individual from the population using roulette wheel selection.
    Each individual is chosen with probability proportional to its fitness.

    Parameters
    ----------
    population : list of list of int
        The current population of tours.
    fitness_vals : np.ndarray
        Array of fitness values corresponding to each individual.

    Returns
    -------
    selected : list of int
        A single tour selected from the population based on fitness.

    Example
    -------
    >>> population = [[0,1,2],[2,1,0],[1,0,2]]
    >>> fitness_vals = np.array([0.1, 0.7, 0.2])
    >>> roulette_select(population, fitness_vals)
    [2,1,0]  # Most likely due to higher fitness
    """

    probs = fitness_vals / np.sum(fitness_vals)
    idx = np.random.choice(len(population), p=probs)
    return population[idx]

def initialize_population(n, pop_size):
    """
    Initialize a population of random tours for the TSP.

    Each tour is a random permutation of city indices.

    Parameters
    ----------
    n : int
        Number of cities.
    pop_size : int
        Number of individuals in the population.

    Returns
    -------
    population : list of list of int
        A list containing `pop_size` tours, each being a permutation of `n` cities.

    Example
    -------
    >>> initialize_population(5, 3)
    [[2, 0, 4, 1, 3], [1, 4, 0, 3, 2], [0, 2, 3, 1, 4]]
    """

    return [random.sample(range(n), n) for _ in range(pop_size)]

def memetic_tsp(cities, pop_size=80, generations=150, mutation_rate=0.02,
    local_search_prob=0.2, elite_fraction=0.3, randomness_in_gx=1):
    
    """
    Solves the Traveling Salesman Problem (TSP) using a **Memetic Algorithm**, 
    which combines a Genetic Algorithm with local search heuristics.

    The algorithm maintains a population of candidate tours, applies selection, 
    crossover, mutation, and optional local search to iteratively improve solutions.

    Algorithm Steps
    ---------------
    1. **Initialization**:
    - Compute the distance matrix for the cities.
    - Initialize a population of random tours.

    2. **Generational Evolution** (repeated for `generations`):
    - Evaluate tour lengths and fitness values.
    - Select elite individuals based on shortest distances.
    - Generate offspring using **greedy stochastic crossover**.
    - Apply **swap mutation** to offspring.
    - With a probability `local_search_prob`, improve offspring using 
        `hybrid_local_search`.
    - Combine elite, offspring, and random survivors to form the new population.

    3. **Termination**:
    - Return the best tour found, its length, and the history of best lengths 
        across generations.

    Parameters
    ----------
    cities : array-like
        Coordinates of the cities as a list or numpy array of shape (n_cities, 2).
    pop_size : int, optional (default=80)
        Number of individuals in the population.
    generations : int, optional (default=150)
        Number of generations to evolve.
    mutation_rate : float, optional (default=0.02)
        Probability of swapping two cities in mutation.
    local_search_prob : float, optional (default=0.2)
        Probability of applying hybrid local search to an offspring.
    elite_fraction : float, optional (default=0.3)
        Fraction of population preserved as elite each generation.
    randomness_in_gx : int, optional (default=1)
        Controls stochasticity in the greedy crossover (top-k nearest selection).

    Returns
    -------
    best_tour : list[int]
        The permutation of city indices representing the best tour found.
    best_length : float
        The total distance of the best tour.
    best_lengths : list[float]
        History of the best tour length in each generation.

    Example
    -------
    >>> cities = np.random.rand(5, 2)
    >>> best_tour, best_len, history = memetic_tsp(cities, pop_size=50, generations=100)
    >>> print(best_tour, best_len)
    [0, 2, 4, 1, 3] 123.45
    """


    n = len(cities)
    dist_matrix = create_distance_matrix(cities)
    population = initialize_population(n, pop_size)

    best_lengths = []

    for gen in range(generations):
        lengths = np.array([total_distance(ind, dist_matrix) for ind in population])
        best_len = lengths.min()
        best_lengths.append(best_len)

        fitness = 1.0 / (lengths + 1e-12)
        elite_size = max(1, int(elite_fraction * pop_size))
        elite_indices = lengths.argsort()[:elite_size]
        elite = [population[i] for i in elite_indices]

        new_pop = []
        while len(new_pop) < pop_size - elite_size:
            p1 = roulette_select(population, fitness)
            p2 = roulette_select(population, fitness)
            child = greedy_crossover_stochastic(p1, p2, dist_matrix, randomness=randomness_in_gx)
            child = mutate_swap(child, mutation_rate)
            if random.random() < local_search_prob:
                child = hybrid_local_search(child, dist_matrix, depth=1)
            new_pop.append(child)

        survivors = []
        if len(new_pop) + elite_size < pop_size:
            survivors = [random.choice(population) for _ in range(pop_size - len(new_pop) - elite_size)]

        population = elite + new_pop + survivors
        population = population[:pop_size]

    final_lengths = [total_distance(ind, dist_matrix) for ind in population]
    best_idx = int(np.argmin(final_lengths))
    return population[best_idx], final_lengths[best_idx], best_lengths
