# Objective functions
reference to Min_Max_Functions.png
A note on Objective functions
• We can minimise or maximise any function whose 
domain is a set with a partial ordering defined on it 
• But, in real cases, we tend to only have functions with a 
scalar real number output 
• Will assume that this is the range of our objective function 
from here on out



# Gradient
Note that the derivative gives 
the gradient or slope of the 
tangent to the curve
• This means, that at an 
optimum, local or global, the 
derivative is 0
• Hence, we can use the 
derivative  to solve for the 
optimum point
• Can use the second order 
derivative to determine if 
maximum or minimum

 Note that we can’t use this in every case 
• Some functions are not continuously differentiable over the feasible set :(  
• Gradient has no analytical solution :(  
• Gradient is expensive to compute :(  (Automatic differentiation can help us with this to some extent 
though) 
• Will encounter many such cases in AI 
• Will learn how to cope 
• Gradient Descent (1st order method) - no analytical solution 
• Metaheuristics (0th order methods) - expensive or non-existent gradient 
• 2nd order methods (Newton-Raphson, qausi-Newton methods) also exist, but are expensive and 
not used that much in AI (yet


# Optimisation = search
• A common theme in my optimisation algorithms is that 
they are essentially searching for the optimal design point 
• This is important to keep in mind when we look at meta
heuristics next class!

# Optimisation Problem Conversion
• Some algorithms are designed to solve minimisation 
problems, other to solve maximisation  
• We can “convert” between the two formulations by negating 
the objective or cost function 
• The solutions of the original and the derived problem are the 
same! 
• Imagine reflecting a graph on the y axis 
• 
argmin f(x) = x* = argmax −f(x)
   x∈𝒳                x∈𝒳

# Optimisation Problem Conversion
• Same thing applies if we add a constant, we shift the 
function up or down, but the solution point remains the 
same 
• 
argmin f(x) = x* = argmin f(x) + c,c ∈ ℝ
     x∈𝒳               x∈𝒳


# Classes of optimisation problem
this has an example png
• There are many ways to classify optimisation problems, 
e.g. convex opimtization, linear, integer, mixed-integer, 
etc… 
• We shall focus on differentiating optimisation problems 
across two dimensions: 
• Unconstrained vs constrained  
• Continuous vs Discrete (Combinatorical)

# Unconstrained vs Constrained 
• Recall the “subject to” portion of the optimisation 
problem formulation 
• We refer to a design point as being an element of a 
feasible set or feasible region 
• So far 
𝒳=dom(f)
, but often times  
𝒳⊂dom(f)
• The former is unconstrained, the latter is constrained

# Constraint-Satisfaction Problems
• Sometimes we don’t care about the value of the objective function or have no 
such functions 
• But we do have constraints 
• We call such instances a CSP 
• Examples: 
• Graph colouring  
• Halls’ marriage problem 
• n-Queens Problem 
• MiniZinc is great at these!

# Continuous vs Discrete
• The characteristics of the objective function’s domain 
impacts the methods we can use greatly 
• Continuous - real numbers, real vectors, real matrices 
• Discrete - integers, integer vectors, integer matrices  
• Many algorithms exploit continuity. So discrete 
optimisation is more difficult :(  
• Integer linear programming is in the class NP

# Problem Conversion
• Sometimes we can convert an  
• Constrained version to an unconstrained version 
• Discrete problem to a continuous problem 
• to approximate a solution 
• Sometimes with provable guarantees :D  
• Won’t look at the guarantees much

# Constrained to unconstrained
• Some methods we will look at will have different ways to 
dealing with converting constrained to unconstrained 
• But in general, we can use penalty methods 
• Basic idea: 
• Keep count of the number of constraints violated 
• multiply by some penalty factor 
• add to cost (minimisation problem) or subtract from 
objective (maximisation problem)

# Gradient Descent (and friends)
• Iterative method for minimisation of convex functions using 
gradient 
• Many improvements, but will look at simple variation for now 
• Start and random answer and iteratively refine answer until 
convergence criteria (usually number of iterations (called 
epochs) is met) 
• Used when solving for root of gradient is not feasible or 
possible  
• The basis of many machine learning algorithms 

## Gradient Descent
• Core idea: gradient of tangent gives us direction of 
steepest ascent 
• Core idea: negation of gradient, should give us direction 
of steepest descent

# Gradient Descent Algorithm

function gd(f, f’, alpha , lo=100, hi=100):
 x = uniform_random(lo, hi)
 gradt = f’(x)
while not converged:
x = x - (alpha)f’(x)
return x


# Gradient Descent - Step Size or Learning Rate

• The step size,α , also called the learning rate can have 
impact on convergence  

• Too large an α, we don’t converge 

• Too small an α, we converge slowly

# Gradient Descent - Vectors or Multivariate
• Gradient Descent is trivially extensible to multivariable or 
vector cases? 
• Just use partial derivative or vector derivate instead! 
• Will see examples in the lab using PyTorch and by hand

# Backtracking
• Can be used to solve CSPs 
• Suppose that we start in state, S0 that is in accord with our constraints but incomplete. We need to take m actions or make m decisions to reach Sm such that we find Sm that obeys our constraints
• We have an action set , A |A| =n of actions that we can take. Our actions are labeled a1, a2, …,an



# Backtracking
. Suppose that we take action a1, and this leads us into state S1 * S1 is a valid state. We now need to move onto S2

•  Suppose that we try all actions and all possible  S3s are invalid
 
• What do we do?


# Backtracking
• No! 
• We assume then that S2 was a bad-turn or dead-end. 
• So we backtrack to S1a nd start from where we left of in our action set  - a1 We now consider a2. And repeat until we either reach a valid Sm and report success or backtrack to S0, exhaust all of our actions and report failure

 

# Backtracking - N Queens
• Consider an n ×nchessboard.  

• We want to find a way to place n queens on it such that 
no queen can attack any other queen 
• Remember, that in chess, a queen can move diagonally, 
horizontally, or vertically any number of squares


# Questions

## Question 1

Let’s formulate a problem!
• Wyndor Glass Ltd. produces two products A and B. A 
makes $2000 of profit per unit, and B makes $3000 of 
profit per unit. There are three plants that manufacture 
three different components that are used to manufacture 
A and B. Plant 1 can only operate for 10 hours per day. 
Plant 2 can only operate for 6 hours a day. Plant 3 can 
only operate for 15 hours a day. Each unit requires 
different amounts of a plants services. These are 
summarised in the following table

## Question 2 is as an image