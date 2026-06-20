import random
import numpy as np

def greedy_crossover_stochastic(p1, p2, dist_matrix, randomness=2):
    
    """
    Generates a child tour (route) by combining two parent tours using a 
    **stochastic greedy crossover** strategy for the Travelling Salesman Problem (TSP).

    This operator maintains the structure of both parents while introducing 
    controlled randomness, helping balance **exploration** (diversity) and 
    **exploitation** (inheritance of good paths).

    Algorithm Steps
    ---------------
    1. Select a random starting city from one of the parent tours.
    2. At each step:
    - Collect neighboring cities (left and right) of the current city from both parents.
    - Filter out already visited cities.
    - Sort the remaining candidates by their distance to the current city.
    - Randomly select the next city from the top `k` nearest candidates, 
        where `k = randomness`.
    3. If no valid neighbors are available, choose the globally nearest remaining city.
    4. Repeat until all cities are included in the child tour.

    Parameters
    ----------
    p1 : list[int]
        The first parent tour (a permutation of city indices).
    p2 : list[int]
        The second parent tour (a permutation of city indices).
    dist_matrix : numpy.ndarray
        A 2D distance matrix where `dist_matrix[i, j]` gives the distance between city `i` and city `j`.
    randomness : int, optional (default=2)
        Controls the degree of randomness in selecting the next city. 
        - Lower values → more greedy behavior (closer cities preferred).
        - Higher values → more stochastic exploration.

    Returns
    -------
    list[int]
        A new child tour (a permutation of city indices) that blends 
        characteristics of both parents.

    Example
    -------
    >>> import numpy as np, random
    >>> random.seed(0)
    >>> p1 = [0, 1, 2, 3, 4]
    >>> p2 = [2, 3, 4, 0, 1]
    >>> dist = np.random.rand(5, 5)
    >>> greedy_crossover_stochastic(p1, p2, dist, randomness=2)
    [3, 4, 0, 1, 2]
    """
    
    n = len(p1)
    remaining = set(p1)
    current = random.choice(p1)
    child = [current]
    remaining.remove(current)

    while remaining:
        candidates = []
        for parent in (p1, p2):
            idx = parent.index(current)
            left = parent[idx - 1]
            right = parent[(idx + 1) % n]
            candidates.extend([left, right])
        candidates = [c for c in candidates if c in remaining]

        if candidates:
            unique = list(dict.fromkeys(candidates))
            sorted_candidates = sorted(unique, key=lambda c: dist_matrix[current, c])
            k = min(randomness, len(sorted_candidates))
            next_city = random.choice(sorted_candidates[:k])
        else:
            next_city = min(list(remaining), key=lambda c: dist_matrix[current, c])

        child.append(next_city)
        remaining.remove(next_city)
        current = next_city

    return child
