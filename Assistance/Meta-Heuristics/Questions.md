# Question 1
A simple example of using a genetic algorithm to optimize a mathematical function. 
Suppose we want to find the maximum value of the function f(x) = x^2 - 4x + 3 over the range x 
∈ [0, 6], where x is a real number (sometimes). We can use a genetic algorithm to find the 
value of x that produces the maximum value of f(x). 

## Step 1: Define the chromosome
Since x is a real number in [0, 6], we use a **real-valued representation**.
Each chromosome is a single real number (one gene) representing a candidate value of x.

Example chromosome: [4.73]

## Step 2: Generate the initial population
Randomly generate a population of N chromosomes (e.g., N = 6) by sampling uniformly from [0, 6].

Example initial population:
| Chromosome | x value |
|---|---|
| C1 | 0.5 |
| C2 | 3.2 |
| C3 | 5.1 |
| C4 | 1.8 |
| C5 | 4.6 |
| C6 | 2.0 |

## Step 3: Evaluate fitness
The fitness function is f(x) = x² - 4x + 3. We evaluate each chromosome:

| Chromosome | x | f(x) = x² - 4x + 3 |
|---|---|---|
| C1 | 0.5 | 0.25 - 2 + 3 = **1.25** |
| C2 | 3.2 | 10.24 - 12.8 + 3 = **0.44** |
| C3 | 5.1 | 26.01 - 20.4 + 3 = **8.61** |
| C4 | 1.8 | 3.24 - 7.2 + 3 = **-0.96** |
| C5 | 4.6 | 21.16 - 18.4 + 3 = **5.76** |
| C6 | 2.0 | 4 - 8 + 3 = **-1.0** |

Since we are maximizing, higher fitness is better. C3 (x=5.1) is the fittest.

## Step 4: Apply genetic operators

### Selection (Tournament Selection, t=3)
Randomly pick 3 chromosomes, take the top 2 as parents.
- Tournament 1: Pick C3(8.61), C1(1.25), C4(-0.96) → **Parents: C3 and C1**
- Tournament 2: Pick C5(5.76), C2(0.44), C6(-1.0) → **Parents: C5 and C2**

### Crossover (Interpolation Crossover, λ = 0.5)
- Offspring 1: (1-0.5)×5.1 + 0.5×0.5 = 2.55 + 0.25 = **2.8**
- Offspring 2: (1-0.5)×4.6 + 0.5×3.2 = 2.3 + 1.6 = **3.9**

### Mutation (Gaussian noise, σ = 0.3)
Add small random Gaussian noise, then clamp to [0, 6]:
- Offspring 1: 2.8 + 0.15 = **2.95**
- Offspring 2: 3.9 - 0.22 = **3.68**

## Step 5: Select the next generation
Combine parents and offspring. Keep the top N by fitness (elitism), or replace worst members.

New population might be: {C3(5.1), C5(4.6), Offspring2(3.68), Offspring1(2.95), C1(0.5), C2(3.2)}

## Step 6: Repeat steps 3-5
Repeat for a fixed number of generations (e.g., 100). Each generation, the population should trend toward higher fitness values as fit individuals are selected and bred.

## Step 7: Output the result
After all generations, return the chromosome with the highest fitness.

The analytical maximum of f(x) = x² - 4x + 3 on [0,6] is at the boundary x = 6:
f(6) = 36 - 24 + 3 = **15**

The GA should converge to x ≈ 6, f(x) ≈ 15.

## Code Solution

```python
import random

def fitness(x):
    return x**2 - 4*x + 3

# Parameters
POP_SIZE = 20
GENERATIONS = 100
X_MIN, X_MAX = 0, 6
MUTATION_RATE = 0.3
MUTATION_SIGMA = 0.5
TOURNAMENT_SIZE = 3

# Step 1 & 2: Generate initial population (real-valued chromosomes)
population = [random.uniform(X_MIN, X_MAX) for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    # Step 3: Evaluate fitness
    scored = [(x, fitness(x)) for x in population]
    scored.sort(key=lambda pair: pair[1], reverse=True)  # Maximization

    new_population = [scored[0][0]]  # Elitism: keep best

    while len(new_population) < POP_SIZE:
        # Step 4a: Tournament selection
        tournament = random.sample(scored, TOURNAMENT_SIZE)
        tournament.sort(key=lambda pair: pair[1], reverse=True)
        parent1 = tournament[0][0]
        parent2 = tournament[1][0]

        # Step 4b: Interpolation crossover
        lam = random.random()
        child = (1 - lam) * parent1 + lam * parent2

        # Step 4c: Mutation (Gaussian noise)
        if random.random() < MUTATION_RATE:
            child += random.gauss(0, MUTATION_SIGMA)
            child = max(X_MIN, min(X_MAX, child))  # Clamp to bounds

        new_population.append(child)

    # Step 5: New generation
    population = new_population

# Step 7: Output result
best = max(population, key=fitness)
print(f"Best x = {best:.4f}")
print(f"f(x)   = {fitness(best):.4f}")
# Expected: x ≈ 6, f(x) ≈ 15
```

---

# Question 2
Simple example of using a genetic algorithm to solve the One Max Problem. 
The One Max Problem is a classic optimization problem where the goal is to find a binary string 
of length N that contains all ones. The fitness of a chromosome is simply the number of ones in 
the string.  

## Step 1: Define the chromosome
We use a **binary array representation**. Each chromosome is a binary string of length N.
Each gene is either 0 or 1.

Example chromosome (N=8): [1, 0, 1, 1, 0, 1, 0, 0]

## Step 2: Generate the initial population
Randomly generate each chromosome by flipping a fair coin for each bit.

Example initial population (N=8, pop_size=4):
| Chromosome | Binary String | Fitness (count of 1s) |
|---|---|---|
| C1 | [1, 0, 1, 1, 0, 1, 0, 0] | 4 |
| C2 | [0, 1, 0, 0, 1, 1, 1, 0] | 4 |
| C3 | [1, 1, 0, 1, 0, 0, 1, 1] | 5 |
| C4 | [0, 0, 1, 0, 1, 0, 0, 1] | 3 |

## Step 3: Evaluate fitness
Fitness = number of 1s in the chromosome. The target fitness is N (all ones).

From above: C3 is the fittest with 5 ones.

## Step 4: Apply genetic operators

### Selection (Roulette Wheel Selection)
Probability of selection proportional to fitness:
- Total fitness = 4 + 4 + 5 + 3 = 16
- P(C1) = 4/16 = 0.25, P(C2) = 4/16 = 0.25, P(C3) = 5/16 = 0.3125, P(C4) = 3/16 = 0.1875

Suppose we select C3 and C1 as parents.

### Crossover (Single-point crossover at position 4)
- Parent 1 (C3): [1, 1, 0, 1, | 0, 0, 1, 1]
- Parent 2 (C1): [1, 0, 1, 1, | 0, 1, 0, 0]
- Offspring 1:    [1, 1, 0, 1, | 0, 1, 0, 0] → fitness = 4
- Offspring 2:    [1, 0, 1, 1, | 0, 0, 1, 1] → fitness = 5

### Mutation (Bit flip, rate = 1/N per bit)
For each bit, flip with probability 1/N:
- Offspring 1: position 5 flips → [1, 1, 0, 1, 0, **0**, 0, 0] → [1, 1, 0, 1, 1, 1, 0, 0] → fitness = 5
- Offspring 2: no mutation → stays [1, 0, 1, 1, 0, 0, 1, 1] → fitness = 5

## Step 5: Select the next generation
Replace the least fit members of the population with the offspring.
New population: {C3(5), Offspring1(5), Offspring2(5), C1(4)}

## Step 6: Repeat steps 3-5
Repeat for multiple generations. The average fitness of the population increases each generation as chromosomes accumulate more 1s.

## Step 7: Output the result
After sufficient generations, the best chromosome should be [1, 1, 1, 1, 1, 1, 1, 1] with fitness = N.

## Code Solution

```python
import random

# Parameters
N = 20               # Length of binary string
POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 1 / N  # Per-bit mutation rate
CROSSOVER_RATE = 0.8

# Step 1: Chromosome = binary list of length N
# Step 2: Generate initial population
population = [[random.randint(0, 1) for _ in range(N)] for _ in range(POP_SIZE)]

def fitness(chromosome):
    """Step 3: Fitness = count of ones"""
    return sum(chromosome)

def roulette_selection(pop, fitnesses):
    """Roulette wheel selection"""
    total = sum(fitnesses)
    if total == 0:
        return random.choice(pop)
    pick = random.uniform(0, total)
    current = 0
    for chrom, fit in zip(pop, fitnesses):
        current += fit
        if current >= pick:
            return chrom[:]
    return pop[-1][:]

def single_point_crossover(p1, p2):
    """Single-point crossover"""
    point = random.randint(1, N - 1)
    child1 = p1[:point] + p2[point:]
    child2 = p2[:point] + p1[point:]
    return child1, child2

def mutate(chromosome):
    """Bit-flip mutation"""
    for i in range(N):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

for gen in range(GENERATIONS):
    fitnesses = [fitness(c) for c in population]
    best_fit = max(fitnesses)

    if best_fit == N:  # Found the optimal solution
        break

    new_population = []
    # Elitism: keep the best
    best_idx = fitnesses.index(best_fit)
    new_population.append(population[best_idx][:])

    while len(new_population) < POP_SIZE:
        # Step 4a: Selection
        parent1 = roulette_selection(population, fitnesses)
        parent2 = roulette_selection(population, fitnesses)

        # Step 4b: Crossover
        if random.random() < CROSSOVER_RATE:
            child1, child2 = single_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1[:], parent2[:]

        # Step 4c: Mutation
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.extend([child1, child2])

    # Step 5: New generation
    population = new_population[:POP_SIZE]

# Step 7: Output
best = max(population, key=fitness)
print(f"Best chromosome: {best}")
print(f"Fitness: {fitness(best)} / {N}")
# Expected: all ones, fitness = N
```

---

# Question 3
Using a genetic algorithm to solve the Traveling Salesman Problem (TSP), which is a classic 
optimization problem in computer science. 
In the TSP, we are given a list of cities and the distances between each pair of cities, and we 
want to find the shortest possible route that visits each city exactly once and returns to the 
starting city. 

## Step 1: Define the chromosome
We use a **permutation representation**. Each chromosome is a permutation of city indices representing the order in which cities are visited.

For 5 cities (0, 1, 2, 3, 4):
Example chromosome: [2, 0, 4, 1, 3] → visit city 2 first, then 0, 4, 1, 3, then return to 2.

This representation guarantees every city is visited exactly once.

## Step 2: Generate the initial population
Randomly shuffle the list of cities to create each chromosome.

Example (5 cities, pop_size=4):
| Chromosome | Route |
|---|---|
| C1 | [0, 3, 1, 4, 2] |
| C2 | [2, 1, 0, 3, 4] |
| C3 | [4, 0, 2, 3, 1] |
| C4 | [1, 2, 4, 0, 3] |

## Step 3: Evaluate fitness
Compute the total distance of the tour (including return to start). Since we want to **minimize** distance, we define fitness as the **negative of the total distance** (or its reciprocal) so that shorter tours have higher fitness.

fitness(chromosome) = -total_distance(chromosome)

Or equivalently: fitness = 1 / total_distance

Example with a distance matrix:
- C1 route: 0→3→1→4→2→0, total distance = 25 → fitness = -25
- C2 route: 2→1→0→3→4→2, total distance = 18 → fitness = -18 (best)

## Step 4: Apply genetic operators

### Selection (Tournament Selection, t=3)
Pick 3 random chromosomes, take the 2 with highest fitness (shortest tour) as parents.

### Crossover (Order Crossover - OX)
Order crossover preserves the relative order of cities, which is critical for permutation-based chromosomes.

Example:
- Parent 1: [0, 3, 1, 4, 2], Parent 2: [2, 1, 0, 3, 4]
- Select a substring from Parent 1 at positions 1-3: [_, **3, 1, 4**, _]
- Fill remaining positions with cities from Parent 2 in order (skip 3,1,4): 2, 0
- Offspring: [2, 3, 1, 4, 0]

### Mutation (Swap Mutation)
Randomly select two positions and swap their cities:
- Before: [2, 3, 1, 4, 0]
- Swap positions 1 and 4: [2, **0**, 1, 4, **3**]
- After: [2, 0, 1, 4, 3]

This maintains the permutation property (every city appears exactly once).

## Step 5: Select the next generation
Use elitism (keep the best tour found so far) and replace the rest of the population with offspring.

## Step 6: Repeat steps 3-5
Repeat for many generations (e.g., 500). The best tour distance should decrease over time as the GA explores and exploits the search space.

## Step 7: Output the result
Return the chromosome (route) with the shortest total distance.

## Code Solution

```python
import random
import math

# Example: 10 cities with random coordinates
NUM_CITIES = 10
POP_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 5

# Generate random city coordinates
random.seed(42)
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)]

def distance(c1, c2):
    """Euclidean distance between two cities"""
    return math.sqrt((cities[c1][0] - cities[c2][0])**2 + (cities[c1][1] - cities[c2][1])**2)

def total_distance(route):
    """Total tour distance (including return to start)"""
    dist = 0
    for i in range(len(route)):
        dist += distance(route[i], route[(i + 1) % len(route)])
    return dist

def fitness(route):
    """Fitness = negative distance (we maximize fitness = minimize distance)"""
    return -total_distance(route)

# Step 1 & 2: Generate initial population (permutation chromosomes)
population = [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POP_SIZE)]

def tournament_selection(pop, k=TOURNAMENT_SIZE):
    """Tournament selection: pick k random, return the best"""
    tournament = random.sample(pop, k)
    return min(tournament, key=total_distance)[:]

def order_crossover(p1, p2):
    """Order Crossover (OX) for permutation chromosomes"""
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copy substring from parent 1
    child[start:end+1] = p1[start:end+1]
    # Fill remaining from parent 2 in order
    p2_filtered = [gene for gene in p2 if gene not in child[start:end+1]]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_filtered[idx]
            idx += 1
    return child

def swap_mutation(route):
    """Swap two random cities in the route"""
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return route

best_ever = min(population, key=total_distance)
best_ever_dist = total_distance(best_ever)

for gen in range(GENERATIONS):
    new_population = [best_ever[:]]  # Elitism

    while len(new_population) < POP_SIZE:
        # Step 4a: Selection
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # Step 4b: Crossover
        child = order_crossover(parent1, parent2)

        # Step 4c: Mutation
        if random.random() < MUTATION_RATE:
            child = swap_mutation(child)

        new_population.append(child)

    # Step 5: New generation
    population = new_population

    # Track best
    current_best = min(population, key=total_distance)
    current_dist = total_distance(current_best)
    if current_dist < best_ever_dist:
        best_ever = current_best[:]
        best_ever_dist = current_dist

# Step 7: Output result
print(f"Best route: {best_ever}")
print(f"Total distance: {best_ever_dist:.2f}")
print(f"City coordinates: {cities}")
``` 

