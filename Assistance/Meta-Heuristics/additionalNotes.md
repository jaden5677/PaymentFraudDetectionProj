# Heuristics
• Sometimes, our problem as a structure that allow us to 
make good guesses about where good regions or good 
design points are located in the search space 
• A rule of thumb for a very specific type of problem is 
called a heuristic 
• Example: When we have different servers at different 
costs for different tasks, we have heuristics that we 
can use to assign servers to tasks

# The problem with heuristics
• Heuristics, however, have problems 
• For one, there are very problem and instance specific 
• Much effort must be applied to derive them 
• Much effort must be applied in their engineering 
• Remember we like algorithms to be as general as 
possible!

# Metaheuristics to the rescue!
• Metaheuristics are like heuristics in that they offer good, 
efficient strategies but often no guarantees of optimality 
• But they differ in that they are more generally applicable 
• Rely on more generic ideas to guide search through space of 
potential solutions 
• No magic bullet, but very helpful 
• Many different types and general ideas 
• Exists an interesting taxonomy over metaheuristics

# The power of metaphor
• Genetic Algorithm and Genetic Programming - Natural 
Selection 
• Particle Swarm Optimisation - flocking behaviour of birds 
and insects 
• Simulated Annealing - the cooling of molten metal into 
solid metal 
• Ant Colony Optimisation - how ants forage for food and 
communicated indirectly by depositing pheromones

# Natural Selection - A very brief summary
• Mechanism by which organisms evolve and develop traits 
• Environment acts as a form of pressure on a population of organisms 
• Organism’s fitness is a result of how well it manages in its environment 
• Organisms that survive mate 
• Fit organisms tend to want to mate with other fit organisms 
• Produce offspring 
• Offspring might acquire some mutation during gestation  
• Offspring then continue the cycle

# A bit of genetics 
• Recall that the features of an organism are partially 
encoded by their genetics  
• Hence, their fitness for their environment is at least 
partially determined by their genetics! 
• DNA and chromosomes comprises of a sequence genes 
that work in concert with one another to encode features


# Key insights of genetic algorithm
• We can represent features (such as the components) of 
our design points as genes 
• Represent the entire  design point as an array of said 
features (like a chromosome) 
• Use chromosome as input into some fitness function that 
assess how good the design point is

# Key insights of genetic algorithm
• When organisms mate, their genetics are melded 
together - so we can combine our artificial chromosomes 
together similarly  
• When organisms mutate, their genes change - can 
change values inside of the array 
• We repeatedly apply this procedure to maximise some 
objective function that acts as our fitness function

# Decisions, decisions, decisions
• The above is an outline 
• We need to decide on some problem specific details 
• Need to define 
• Representation 
• Crossover operation 
• Mutation 
• And of course fitness


# Representation of Chromosomes
• We need to design an encoding of our design points as 
arrays.  
• Each problem is different 
• Hence would have to select our representation to suite 
• Choosing a good representation is vital 
• Our choice of representation also dictates how we will 
randomly initialise our population 
• And how we apply other operations

# Representation of Chromosomes
• Three main ways: 
• Real vectors - used for problems over real vector 
space. Each gene is component of vector is a real 
number representing a component 
• Binary arrays - each entry is 0 or 1 representing true or 
false for some feature associated with that entry. These 
can also be used to represent integer values 
• Permutations - the entire chromosome is a permutation 


# Selection strategy
• Roulette Wheel selection - all valid chromosomes in a 
population are considered for mating. Parents are 
selected stochastically in proportion to their fitness value 

• Tournament Selection - t chromosomes are randomly 
selected. The top 2 are taken and then mated  

• Truncation Selection - pick top t chromosomes, randomly 
choose 2 and mate them


# Crossover Methods
• Different methods of crossover 
• Single-point crossover 
• Multiple-point crossover 
• Uniform crossover 
• Interpolation crossover


# Interpolation Crossover
• Consider there being a line between two vectors x1 and x2. This line is defined by the expression 
(1 −λ)x1 + λx2 where 
0 ≤λ≤1


• We can look at crossover as choosing a point on this line. 
• We either randomly select a  λ

• Or set λ to a fixed value (usually 0.5) 
• Interpolation crossover can only be done on chromosomes that 
are real vectors

# Mutation
• Mutation can take on many 
different forms depending on 
the nature the representation
• Recall that our chromosome 
can be conceived as existing in 
some space of potential 
chromosomes
• Mutation makes this 
chromosome randomly jump to 
another neighbouring solution
• Mutation helps us avoid getting 
stuck in a bad local minimum 
# Mutation

• Possible interpretations 
• Adding standard Gaussian noise to a real vector 
chromosome 
• Flipping a bit in a bit-based chromosome 
• Swapping elements in a permutation based 
chromosome

# GA Usage
• Using GAs require a deal of thought 
• But they tend to work well when our search space is 
high, but number of feasible points is low 
• Examples: 
• Scheduling  
• Automated design


# Particle Swarm Optimisation
• Based off of flocking behaviour of birds and 
insects
• “Follow” the leader type strategy
• Each particle has a memory of its previous 
best state
• Each solution represented as a particle
• Usually a real vector
• At the end of every iteration
• Compute a velocity of each particle that 
gives it a direction to travel in
• Add velocity to particle to compute new 
direction of particle
• Unlike GA, we can naturally use PSO for 
both maximisation and minimisation 
problems

# Velocity Computation
• To decide on a particle’s velocity, we take into account 
three pieces of information 
• Inertia - the resistance to change velocity 
• Exploration factor - tendency to explore the current 
globally best behaviour  
• Exploitation factor - tendency to follow previously 
known best personal behaviour

# PSO Notation

x^(i) is the ith particle  

v^(i) is the velocity of the ith particle 

x^(i)v(best) is the best state of particle i in its entire history  

xv(best) is the state of the best performing particle over all iterations

w - inertial factor, usually set to a value between 0 and 1

r1 -exploitation factor, randomly choose each time from  𝒰(0,1)

• r2 - exploration factor, randomly choose each time from 𝒰(0,1)

function pso( , , , num_iter): 
    population = [] 
    p_best = None 
    best_score = -inf  
    for i = 1 to k do 
        population.append(init_particle()) 
    for i = 1 to num_iter:  
        for p in population do 
            f_p = f(p.state) 
            if f_p > p.prev_best_score: 
                p.prev_best = p.state 
            if f_p > best_score: 
                best_score = f_p 
                p_best = p.state 
        for p in population do:  
             = random_uniform(0, 1) 
             = random_uniform(0, 1) 
            velocity =  * p.prev_velocity + (p.prev_best - p.state) + (p_best - p.state) 
            p.state = p.state + velocity 
            p.prev_velocity = velocity 
    return p_best 


Hill-Climbing Techniques
• GA and PSO are what are referred to as population 
based meta-heuristics  
• They generate a random population, and use them in 
concert with one another to find a good design point 
• Some methods, however, instead focus on randomly 
generating a single design point and iteratively refining it 
• Hill-climbing techniques are a family of such techniques


• Core idea, start at random point
• Each point has neighbouring 
points
• Examine some neighbouring 
points and test if they are better 
than our current point
• If better move to that point
• If equal or worse stay where 
we are
• continue until converge 
criteria met


• There is a major problem with vanilla hill-climbing 
techniques 
• Vanilla hill-climbing techniques, more formally called 
deterministic hill climbing techniques can get a stuck in a 
bad local optimal point 
• Sometimes, we need to get worse before we get better 
• Stochastic hill-climbing techniques avoid this problem by 
computing some probability of moving to a neighbouring 
point, and performing a Bernouilli Trial 

# Simulated Annealing
• SA is a variant of stochastic hill climbing 
• Inspired by the process of smelting metal 
• We provide a temperature schedule that controls volatility 
or willingness to accept a worse point in hopes of going 
to a better point in the future   
• As time goes on (our iterations increase), temperature 
falls, thereby lowering volatility

# Simulated Annealing - Probabilistic Jumping - reference images

# Simulated Annealing - Temperature Schedules
•  Different types of temperature schedules 
• Fixed schedules - we come up with temperature 
values before hand 
• Non-fixed schedules - give starting temperature and 
parameters of schedule and temperature is 
dynamically calculated
see image

# Simulated annealing function

function fimulated_annealing(f,x,T,t, k_max)
    y = f(x)
    x_best, y_best = x,y
    for k in l: kmax
        x' = x + rand(T)
        y' = f(x')
        dy = y' - y
        if dy <= 0 || rand() < exp(-dy/t(k))
            x,y = x',y'
        end
        if y' < y_best
            x_best, y_best = x', y'
        end
    end
    return x_best
end

