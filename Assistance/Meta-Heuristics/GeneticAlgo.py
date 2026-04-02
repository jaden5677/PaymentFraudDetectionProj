#Genetic Algorithms These are all example functions to show what is expected



Recall from lectures the Genetic Algorithm's flow chart below
!["GA Flow Chart"](https://github.com/justkrismanohar/comp3608-2025-labs/blob/main/Lab4/GA_flowchart.jpg?raw=true)

There are several parameters for our ***natural selection*** process that need to be defined first.

We can start with easiest the hyper-parameters
$k - \text{ the population size}\\ e - \text{ the top $e$ members to move on from one population to the next}\\ p_c - \text{ the probability of crossover} \\ p_m - \text{ the probability of mutation}$

These values can be tweaked over serveral runs of the genetic algorithm.

The other parts of the ***natural selection*** methaphor are a bit harder to pin down, exactly right, and often would require a great deal of experimentation to get right based on the problem.

## Selection
The first consideration is how parents are selected to populate the next generation and move the silumation forward. Two strategies that seem to work well in practise are ```RouletteWheelSelector``` and ```TournamentSelection```.

Below are some barebones impelmentations of selection mechanisms used in genetic algorithms. In practice these would likely be implemented more efficitently by a library, however as good sceince studnets it important to have an idea of how it can be done from scratch 😉

```Python
import numpy as np

def top_k(arr, k):
    return np.argpartition(arr, -k)[:k]

def elitism(e, population, fitnesses):
    top = top_k(fitnesses, e)
    return population[top]

class Selector(object):
    def __init__(self):
        super().__init__()

    def prep_selector(self, fitnesses):
        pass

    def select(self, population):
        pass


class RouletteWheelSelector(Selector):
    def __init__(self):
        super().__init__()
        self.table = []
    
    def prep_selector(self, fitnesses):
        transformed = np.exp(fitnesses)
        total = np.sum(transformed)
        self.table = transformed / total
        self.indicies = np.arange(0, len(fitnesses), 1, dtype=int)

    def select(self, population):
        parent1_idx, parent2_idx = np.random.choice(self.indicies, size=2, replace=False, p=self.table)
        parent1 = population[parent1_idx, :]
        parent2 = population[parent2_idx, :]
        return parent1, parent2

class TournamentSelection(Selector):
    def __init__(self):
        super().__init__()
        self.table = []
    
    def prep_selector(self, fitnesses):
        self.table = fitnesses

    def select(self,population):
        indicies = np.arange(0, len(population), 1, dtype=int)
        sub_population = population[indicies]
        sub_pop_fitnesses = self.table[indicies]
        parent1_idx, parent2_idx = top_k(sub_pop_fitnesses, 2)
        return sub_population[parent1_idx], sub_population[parent2_idx]
```

## Crossover
After two parents have been selected to the other process of concern is producing a genetic off spring. Here, the focus is primarily on how are these genees to be exchange in such a way that they can ***exploit*** the things that work but also leave some room to ***explore*** other possible soultions.

Again, below is just one way you can do it from scratch. Here we have a barebones implementation of ```single_point```, ```interpolation``` and ```uniform``` crossovers dicussed in lecture.

```Python
import numpy as np


def single_point(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.zeros_like(parent1)
    child[:point] = parent1[:point]
    child[point:] = parent2[point:]
    return child


def interpolation(parent1, parent2):
    lambda_point = np.random.uniform()
    return lambda_point * parent1 +  (1 - lambda_point) * parent2

def uniform(parent1, parent2):
    selection = np.random.randint(0, 2, size=len(parent1))
    neg_selection = 1 - selection
    child = parent1 * selection + parent2 * neg_selection
    return child
```


## The Genetic Algorithm 🧬
So now we're almost ready to impelemnt our natural selection methaphor.

The remaing parts that are missing are a ```fitness(x)``` function that evaluates how well a specific chromosome ```x``` performs and a ```mutate(x)``` that ***randomdly*** modifies the genes of ```x``` in some where. Typically, the detials of these process are heavily problem specific, more so that the crossover and selection processes defined above.

The ```fitness``` can conduct a simple evaluation procedure over ```x``` via ```for``` loop or execut a complex simulation to see how well the ***features*** of expressed by the ***genes*** of ```x``` perfrom in the desired environment. Similarly  ```mutate(x)``` determine which genes to modify and how can range from simple to complex. These two aspects require alot attention because they heavily infulence how the natural selection methaphor will ***exploit*** \(i.e., ```fitness```\) and ***explore*** \(i.e., ```mutate(x)```\) genetic solutions.  

Assuming these two functions exisit for a given problem we can now implement our barebones version of the ***natural selection*** process i.e., over very own genetic algorithm 🧬 ☢ 🤓

## ⚠⚠⚠⚠ Ensure you can follow the parts of the code below and re-create it / modify from scratch ⚠⚠⚠⚠


```Python
def mutate(x):
  pass

def fitness(x):
  pass

## Store hyperparamters:
p_c = 0.8
p_m = 0.1
k = 100
e = 5
num_iter = 100

population = #generate initial genetic population of size k
fitnesses = np.array([fitness(x) for x in population])
selector = RouletteWheelSelector()

for i in range(num_iter):
    convergence_curve.append(max(fitnesses))
    new_population = elitism(e, population, fitnesses)
    selector.prep_selector(fitnesses)
    while len(new_population) < k:
        parent1, parent2 = selector.select(population)
        perform_crossover = np.random.random() <= p_c
        if perform_crossover:
            child = single_point(parent1, parent2)
            perform_mutation = np.random.random() <= p_m
            if perform_mutation:
                child = mutate(child)
            new_population = np.append(new_population, [child], axis=0)
    population = new_population
    fitnesses = np.array([fitness(x) for x in population])
```

## Let's store the data for the problem
n = 100
p = np.random.randint(5, 8, size=n)
w = np.random.randint(10, 45, size=n)
C = 365

## Specific for this problem
## Below are some of the parts that will vary based on the
## problem representation
def compute_penalty(x):
    total_weight = np.dot(w, x)
    #Observe what happens when total_weight is <, >, or =,
    #to the capacity C
    penalty = np.maximum(0, -C + total_weight)
    return penalty
    
def fitness(x):
    profit = np.dot(p, x)
    penalty = compute_penalty(x) # we apply a penalty for how much extra weight above capacity a solution incurs
    return profit - penalty

def mutate(x):
    point = np.random.randint(1, len(x))
    x[point] = 1 - x[point]
    return x

## Store hyperparamters:
p_c = 0.8
p_m = 0.1
k = 100
e = 5
num_iter = 100

## For visulizing the fitness overtime so we can estimate
## the GA's performance
convergence_curve = []

## Note how we initalize the genetic population of size k
## Remember this is always problem specific
population = np.random.randint(0, 2, size=(k, n))

fitnesses = np.array([fitness(x) for x in population])
selector = RouletteWheelSelector()

for i in range(num_iter):
    convergence_curve.append(max(fitnesses))
    new_population = elitism(e, population, fitnesses)
    selector.prep_selector(fitnesses)
    while len(new_population) < k:
        parent1, parent2 = selector.select(population)
        perform_crossover = np.random.random() <= p_c
        if perform_crossover:
            child = single_point(parent1, parent2)
            perform_mutation = np.random.random() <= p_m
            if perform_mutation:
                child = mutate(child)
            new_population = np.append(new_population, [child], axis=0)
    population = new_population
    fitnesses = np.array([fitness(x) for x in population])


import numpy as np

def top_k(arr, k):
    return np.argpartition(arr, -k)[-k:]

def elitism(e, population, fitnesses):
    top = top_k(fitnesses, e)
    return population[top]

class Selector(object):
    def __init__(self):
        super().__init__()

    def prep_selector(self, fitnesses):
        pass

    def select(self, population):
        pass


class RouletteWheelSelector(Selector):
    def __init__(self):
        super().__init__()
        self.table = []

    def prep_selector(self, fitnesses):
        transformed = np.exp(fitnesses)
        total = np.sum(transformed)
        self.table = transformed / total
        self.indicies = np.arange(0, len(fitnesses), 1, dtype=int)

    def select(self, population):
        parent1_idx, parent2_idx = np.random.choice(self.indicies, size=2, replace=False, p=self.table)
        parent1 = population[parent1_idx, :]
        parent2 = population[parent2_idx, :]
        return parent1, parent2

class TournamentSelection(Selector):
    def __init__(self):
        super().__init__()
        self.table = []

    def prep_selector(self, fitnesses):
        self.table = fitnesses

    def select(self,population):
        indicies = np.arange(0, len(population), 1, dtype=int)
        sub_population = population[indicies]
        sub_pop_fitnesses = self.table[indicies]
        parent1_idx, parent2_idx = top_k(sub_pop_fitnesses, 2)
        return sub_population[parent1_idx], sub_population[parent2_idx]


def single_point(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child = np.zeros_like(parent1)
    child[:point] = parent1[:point]
    child[point:] = parent2[point:]
    return child


def interpolation(parent1, parent2):
    lambda_point = np.random.uniform()
    return lambda_point * parent1 +  (1 - lambda_point) * parent2

def uniform(parent1, parent2):
    selection = np.random.randint(0, 2, size=len(parent1))
    neg_selection = 1 - selection
    child = parent1 * selection + parent2 * neg_selection
    return child


import matplotlib.pyplot as plt
plt.plot(list(range(len(convergence_curve))), convergence_curve)
best = elitism(1, population, fitnesses)[0]
best_fit = fitness(best)
best_penalty = compute_penalty(best)
print("best fitness ", best_fit," best_penalty ", best_penalty)
print("best genetic solution ", best)