import random

def mutate_swap(ind, mutation_rate=0.2):
    
    """
    Perform a swap mutation on a tour with a given probability.

    A subsequence of the tour is selected randomly and reversed to introduce variation.

    Parameters
    ----------
    ind : list of int
        The tour (permutation of cities) to be mutated.
    mutation_rate : float, optional
        Probability of applying the mutation (default is 0.2).

    Returns
    -------
    mutated_ind : list of int
        The possibly mutated tour.

    Example
    -------
    >>> ind = [0, 1, 2, 3, 4]
    >>> mutate_swap(ind, mutation_rate=1.0)
    [0, 3, 2, 1, 4]  # Example output with a reversed subsequence
    """

    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(ind)), 2))
        ind[i:j+1] = reversed(ind[i:j+1])
    return ind
