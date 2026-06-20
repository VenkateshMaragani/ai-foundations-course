
TSP - Memetic Algorithm:

    A Memetic Algorithm (MA) implementation to solve the Traveling Salesman Problem (TSP) — combining Genetic Algorithms and local search heuristics for optimized route planning.

Overview:

    This project solves the Traveling Salesman Problem (TSP) using a Memetic Algorithm, which enhances the traditional Genetic Algorithm (GA) by including local optimization steps.
    It aims to find the shortest possible route that visits all cities exactly once and returns to the starting city.

Key Features:

    Greedy-Stochastic Crossover (smart, distance-based crossover)
    Mutation for route diversification
    Local optimization for fine-tuning best routes
    Multiple runs with convergence visualization
    Automatic result plots and summaries

Algorithm Workflow:

flowchart TD
    A[Start] --> B[Initialize Population]
    B --> C[Evaluate Fitness]
    C --> D[Select Parents]
    D --> E[Greedy Stochastic Crossover]
    E --> F[Mutation]
    F --> G[Local Search Optimization]
    G --> H[Evaluate Offspring]
    H --> I[Replace Population]
    I --> J{Stopping Criteria?}
    J -- No --> D
    J -- Yes --> K[Best Route Found]
    K --> L[Save and Plot Results]

Crossover Used — Greedy Stochastic Crossover:

    This crossover intelligently combines two parent routes (p1, p2) to produce a high-quality child.

    Steps:

        1) Pick a random starting city.
        2) Collect neighbors (left/right) from both parents.
        3) Filter only unused cities.
        4) Sort neighbors by distance from current city.
        5) Randomly choose from top-k closest (adds diversity).
        6) If no candidates left → choose globally nearest remaining city.
        7) Repeat until all cities are added.

    Results & Visualizations

    After running, all results are saved automatically in the results/ folder.

Installation & Setup:

    1) Create and activate a virtual environment

    python3 -m venv .venv
    source .venv/bin/activate --> Linux/Mac
    .venv\Scripts\activate    --> Windows

    2) Install dependencies

    pip install -r requirements.txt

    3) Run the main program

    python main.py or python3 main.py


Project Structure:

    TSP_Memetic_Algorithm-main/
    ├── main.py                    # Main script to run the Memetic Algorithm
    ├── berlin52.tsp               # Benchmark dataset file
    ├── README.md                  # Project documentation
    ├── requirements.txt           # Dependencies
    ├── results/                   # Folder to store output plots and results
    │   └── Berlin52_results_run_.png
    └── tsp/                       # Python package containing TSP modules
        ├── __init__.py
        ├── datasets.py            # Dataset loading functions (Berlin52, ATT532, etc.)
        ├── distance_fun.py        # Distance matrix and total distance calculations
        ├── experiments.py         # Experiment runner for multiple algorithm runs
        ├── greedy_crossover.py    # Greedy crossover implementation
        ├── hybrid_ls.py           # Hybrid local search (2-opt + Lin-Kernighan)
        ├── memetic_algo.py        # Memetic Algorithm implementation
        ├── mutation.py            # Mutation functions
        └── plots.py               # Plotting functions for results



Example Output (Berlin52 Dataset):

    Inverse Mutation
    Runs: 30
    Population Size: 150
    Generations Size: 500
    Mutation Rate : 0.15
    Local Search Probability : 0.15
    Elite Fraction: 0.15
    Randomness in Crossover: 1

    Run 1/30: best_len=7544.37, time=11.729s
    Run 2/30: best_len=7544.37, time=11.642s
    Run 3/30: best_len=7658.96, time=11.528s
    Run 4/30: best_len=7658.96, time=11.514s
    Run 5/30: best_len=7544.37, time=11.525s
    Run 6/30: best_len=7544.37, time=11.655s
    Run 7/30: best_len=7544.37, time=11.561s
    Run 8/30: best_len=7544.37, time=11.500s
    Run 9/30: best_len=7544.37, time=11.511s
    Run 10/30: best_len=7548.99, time=11.511s
    Run 11/30: best_len=7544.37, time=11.573s
    Run 12/30: best_len=7544.37, time=11.575s
    Run 13/30: best_len=7544.37, time=11.693s
    Run 14/30: best_len=7544.37, time=11.545s
    Run 15/30: best_len=7544.37, time=11.555s
    Run 16/30: best_len=7544.37, time=11.494s
    Run 17/30: best_len=7544.37, time=11.531s
    Run 18/30: best_len=7548.99, time=11.529s
    Run 19/30: best_len=7658.96, time=11.575s
    Run 20/30: best_len=7544.37, time=11.552s
    Run 21/30: best_len=7548.99, time=11.552s
    Run 22/30: best_len=7548.99, time=11.570s
    Run 23/30: best_len=7548.99, time=11.585s
    Run 24/30: best_len=7544.37, time=11.554s
    Run 25/30: best_len=7544.37, time=11.566s
    Run 26/30: best_len=7544.37, time=11.603s
    Run 27/30: best_len=7544.37, time=11.539s
    Run 28/30: best_len=7544.37, time=11.515s
    Run 29/30: best_len=7544.37, time=11.561s
    Run 30/30: best_len=7544.37, time=11.528s

    === Summary ===
    Runs: 30
    Avg Length: 7556.60
    Best Length: 7544.37
    Avg Runtime: 11.562s\n
    Optimum: 7542, Error: 0.03%
    Saved plot results/Berlin52_results_run_.png
