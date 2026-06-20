import random
from .distance_fun import total_distance

def two_opt_once(tour, dist_matrix, attempts=10):
    
    """
    Performs a **single 2-opt local search** improvement on a given TSP tour.

    The 2-opt algorithm attempts to improve a route by reversing the order 
    of a segment of cities, which can potentially reduce the total distance 
    of the tour. This function performs a limited number of random swap attempts 
    and applies the first improving move it finds.

    Algorithm Steps
    ---------------
    1. Start with the given tour as the current best solution.
    2. For a fixed number of `attempts`:
    - Randomly select two distinct indices `i` and `j` (ensuring they are not adjacent).
    - Reverse the order of cities between `i` and `j`.
    - Compute the total distance of the new tour.
    - If the new tour has a shorter distance, return it immediately.
    3. If no improvement is found after all attempts, return the original tour.

    Parameters
    ----------
    tour : list[int]
        The current tour (a permutation of city indices).
    dist_matrix : numpy.ndarray
        A 2D matrix of pairwise distances between cities.
    attempts : int, optional (default=10)
        Number of random swap attempts to try for finding an improved route.

    Returns
    -------
    list[int]
        An improved tour if a better 2-opt swap is found; 
        otherwise, the original tour is returned.

    Example
    -------
    >>> import numpy as np, random
    >>> from distance_fun import total_distance
    >>> random.seed(0)
    >>> dist = np.random.rand(5, 5)
    >>> tour = [0, 1, 2, 3, 4]
    >>> two_opt_once(tour, dist)
    [0, 1, 3, 2, 4]
    """

    best = tour[:]
    best_len = total_distance(best, dist_matrix)
    n = len(tour)
    for _ in range(attempts):
        i, j = sorted(random.sample(range(n), 2))
        if j - i <= 1:
            continue
        new_tour = best[:i] + best[i:j][::-1] + best[j:]
        new_len = total_distance(new_tour, dist_matrix)
        if new_len < best_len:
            return new_tour
    return best

def lin_kernighan_once(tour, dist_matrix, attempts=5):
    """
    Performs a **single Lin–Kernighan-style local search** iteration on a given TSP tour.

    This function applies a simplified version of the Lin–Kernighan heuristic, 
    which extends the 2-opt approach by allowing multiple segment reversals 
    to explore more complex route improvements. It performs a limited number of 
    random triple-segment reversals to find a shorter tour.

    Algorithm Steps
    ---------------
    1. Copy the given tour and calculate its total distance.
    2. For a fixed number of `attempts`:
    - Randomly select three distinct indices `a`, `b`, and `c`, ensuring order (`a < b < c`).
    - Reverse the tour segments `[a:b]` and `[b:c]`, generating a new tour candidate.
    - Compute the total distance of the new tour.
    - If the new tour has a shorter distance, return it immediately.
    3. If no improvement is found, return the original tour.

    Parameters
    ----------
    tour : list[int]
        The current tour (a permutation of city indices).
    dist_matrix : numpy.ndarray
        A 2D array containing pairwise distances between cities.
    attempts : int, optional (default=5)
        Number of random triple-segment reversals to attempt for improvement.

    Returns
    -------
    list[int]
        An improved tour if a shorter one is found; otherwise, the original tour.

    Example
    -------
    >>> import numpy as np, random
    >>> from distance_fun import total_distance
    >>> random.seed(0)
    >>> dist = np.random.rand(5, 5)
    >>> tour = [0, 1, 2, 3, 4]
    >>> lin_kernighan_once(tour, dist)
    [0, 2, 1, 3, 4]
    """

    best = tour[:]
    best_len = total_distance(best, dist_matrix)
    n = len(tour)
    for _ in range(attempts):
        a, b, c = sorted(random.sample(range(n), 3))
        new_tour = best[:a] + best[a:b][::-1] + best[b:c][::-1] + best[c:]
        new_len = total_distance(new_tour, dist_matrix)
        if new_len < best_len:
            return new_tour
    return best

def hybrid_local_search(tour, dist_matrix, depth=2):
    
    """
    Applies a **hybrid local search** that combines 2-opt and Lin–Kernighan heuristics 
    to iteratively improve a given TSP tour.

    This function alternates between 2-opt and Lin–Kernighan local improvements 
    for a specified number of iterations (`depth`). The combination allows 
    exploration of both small (2-opt) and more complex (Lin–Kernighan) neighborhood 
    structures to enhance solution quality.

    Algorithm Steps
    ---------------
    1. Start with the given tour as the initial solution.
    2. Repeat for a given number of `depth` iterations:
    - Apply a 2-opt move to reduce route length by swapping two cities.
    - Apply a Lin–Kernighan move to explore deeper improvements via multiple segment reversals.
    3. Return the best improved tour found.

    Parameters
    ----------
    tour : list[int]
        The initial tour (a permutation of city indices).
    dist_matrix : numpy.ndarray
        A 2D matrix containing pairwise distances between cities.
    depth : int, optional (default=2)
        Number of hybrid improvement iterations to perform.

    Returns
    -------
    list[int]
        The improved tour after applying hybrid local search.

    Example
    -------
    >>> import numpy as np
    >>> from distance_fun import create_distance_matrix
    >>> cities = np.random.rand(5, 2)
    >>> dist = create_distance_matrix(cities)
    >>> tour = [0, 1, 2, 3, 4]
    >>> hybrid_local_search(tour, dist, depth=2)
    [0, 2, 1, 3, 4]
    """

    improved = tour[:]
    for _ in range(depth):
        improved = two_opt_once(improved, dist_matrix)
        improved = lin_kernighan_once(improved, dist_matrix)
    return improved
