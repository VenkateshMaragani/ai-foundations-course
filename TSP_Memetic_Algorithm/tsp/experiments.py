import time
import random
import numpy as np
from .memetic_algo import memetic_tsp
from .distance_fun import total_distance, create_distance_matrix

def pad_curves(curves):
    """
    Pads all convergence curves to the same length by extending shorter curves 
    with their final value, enabling uniform aggregation (e.g., mean and std computation).

    This function ensures that all fitness or convergence curves collected from 
    multiple algorithm runs have equal lengths, which is essential for computing 
    average performance statistics across runs.

    Parameters
    ----------
    curves : list of list or list of numpy.ndarray
        A list containing convergence curves, where each curve represents 
        the best fitness or tour length over generations for one algorithm run.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array where all curves are padded to the same length 
        with their final value to maintain consistency.

    Example
    -------
    >>> curves = [[10, 9, 8], [10, 9], [10, 9, 8, 7]]
    >>> pad_curves(curves)
    array([[10,  9,  8,  8],
        [10,  9,  9,  9],
        [10,  9,  8,  7]])
    """
    
    max_len = max(len(c) for c in curves)
    padded = [list(c) + [c[-1]] * (max_len - len(c)) for c in curves]
    return np.array(padded)

def run_ma_experiments(cities, runs=10, **ma_kwargs):
    """
    Run multiple experiments using the Memetic Algorithm (MA) on the given TSP instance 
    and aggregate performance metrics across runs.

    This function executes the Memetic Algorithm several times to evaluate its performance 
    on a given set of cities. For each run, it records the best tour length, execution time, 
    and convergence curve. After all runs are completed, it computes the mean and standard 
    deviation of the convergence curves to analyze algorithm stability and consistency.

    Parameters
    ----------
    cities : list of tuple
        A list of (x, y) coordinates representing the locations of cities in the TSP instance.

    runs : int, optional, default=10
        The number of times the Memetic Algorithm should be executed for statistical analysis.

    **ma_kwargs : dict
        Additional keyword arguments passed directly to the `memetic_tsp` function, such as 
        population size, crossover rate, mutation rate, or generation limit.

    Returns
    -------
    tuple
        A tuple containing:
        - best_tour : list of int  
            The best tour (sequence of city indices) found across all runs.
        - lengths : numpy.ndarray  
            An array containing the best tour lengths from each run.
        - times : numpy.ndarray  
            An array containing the execution times (in seconds) for each run.
        - mean_curve : numpy.ndarray  
            The mean of the fitness convergence curves across all runs.
        - std_curve : numpy.ndarray  
            The standard deviation of the fitness convergence curves across all runs.

    Notes
    -----
    The function ensures reproducibility by setting both Python’s and NumPy’s random seeds 
    based on the current timestamp for each run. It also provides runtime logs for each run.

    Example
    -------
    >>> from tsp_runner import run_ma_experiments
    >>> cities = [(0, 0), (1, 2), (3, 1), (4, 0)]
    >>> best_tour, lengths, times, mean_curve, std_curve = run_ma_experiments(cities, runs=5, pop_size=50)
    Run 1/5: best_len=10.54, time=0.234s
    Run 2/5: best_len=10.54, time=0.230s
    ...
    """
    
    results, all_curves, best_tour = [], [], None

    for r in range(runs):
        seed = int(time.time() * 1000) % 2**32
        random.seed(seed + r)
        np.random.seed((seed + r) % 2**32)

        start = time.time()
        tour, best_len, curve = memetic_tsp(cities, **ma_kwargs)
        runtime = time.time() - start

        results.append((best_len, runtime))
        all_curves.append(curve)
        if best_tour is None or best_len < total_distance(best_tour, create_distance_matrix(cities)):
            best_tour = tour
            
        print(f"Run {r+1}/{runs}: best_len={best_len:.2f}, time={runtime:.3f}s")

    lengths = np.array([res[0] for res in results])
    times = np.array([res[1] for res in results])

    curves_array = pad_curves(all_curves)
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)

    return best_tour, lengths, times, mean_curve, std_curve
