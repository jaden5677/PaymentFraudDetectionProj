# Lecture #4b: Maximum Likelihood Estimation and Fitting Probability Distributions

# Uses of optimisation in achieving rational agents
• Recall our definition of rational agents 
• Rational agents try to optimise some performance measure  
• Optimisation is critical both directly and indirectly to achieving 
rational behaviour from agents 
• So far, we looked at how optimisation helps with the problem 
directly 
• Now we look at indirect case that is monumentally 
important for when we tackle ML in a few lectures down 
the road

# Probability and Intelligence
• Many sources of uncertainty  
• Hence, all models of the world carry a degree of uncertainty 
• Hence, the world can be understood through the help of the language of 
uncertainty - probability! 
• We make judicious use of probability distributions to help model the world 
• Will assume that you retained some basic intuition and knowledge of probability 
• Basic axioms, definition of a PMF, PDF, and independence, intuition of what 
a random variable is, and basic notions of a probability distribution  
• Will skip Bayesian stuff for now, but it will become important at the last topic 
of the course, so start looking over from now :)

# Frequentism vs Bayesianism
• Two competing definition on the 
“meaning” behind probability
• In the frequentism case, we assume 
probabilities describe the frequency 
of events
• In the bayesian case, we assume 
probabilities are meant to express 
degrees of belief
• Subtle differences that can lead to 
different methodological approaches
• We will look at a Frequentism 
approach today, bayesian 
approaches towards the end to the 
course

# Probability Distribution
• Have some random variable that can take on many different values 
• Discrete and continuous cases 
• Probability of different values, can be described by probability mass functions 
(discrete) or proportionally described by a probability density function 
• Several common patterns emerge in terms.  
• General Patterns that be refined in terms of using several parameters 
describe the shape of the distribution 
• Each pattern has common use cases 
• Usually we assume a particular pattern when dealing with some data

# Probability Distributions Notation and Poisson Distribution are in images in this folder

# Fitting a distribution
• Sometimes, we can make a reasonable guess or 
assumption about what distribution our data follows 
• … but we don’t know the parameters of the distribution! 
• To meaningfully compute things or answer questions, we 
need to know the parameters as well! 
• How to compute parameters
• Think of the distribution like an 
article of clothing
• Just like a tailor/seamstress can 
measure you to create the best 
suit or dress that we can fit you, 
we can do the same with data
• We need to find the optimal fit
• We need to measure the degree 
to which a distribution with a 
particular configuration of 
parameters fits the data
• This can be used as our 
objective function

# Likelihood function
• Suppose that we have a set of data points, 
D={x<sub>1</sub>,x<sub>2</sub>,…,x<sub>n</sub>}
• And we are trying to measure the fit of a distribution 
  
PD(θ1,θ2,…,θm) to D
• How to do this? 
• We still need to make a few assumptions 
• Main assumption i.i.d - independent and identically 
distributed 

# I.I.D Assumption
• Two components: 
• Identically - the same.  
• Assume that all data points are drawn from the same 
distribution with the same parameters 
• Independent - the probability of two data points are 
independent, 
P(x1, x2|θ1, θ2, …,θm) = P(x1|θ1,θ2,…,θm)P(x2|θ1,θ2,…,θm)


Deriving the likelihood function
• The likelihood function measures how probable our data 
is under our probability distribution 
• Hence, our Likelihood function is the product of the 
probability of each data point under the model (i.i.d) 
assumption

## IID assumption formula in images

# Example
• Suppose that I have a coin (that may not be fair), and I let 
the flip coming up heads as a success 
• When we have two outcomes, a “success” and a 
“failure”, we model using a Bernoulli distribution 
• Assume that we flip the coin 5 times and get the 
following sequence of results: HHTTH. 
• What is the probability of success (i.e. the parameter of 
the Bernoulli distribution)?

ℒ(HHTTH,θ) = P(HHTTH|θ)
P(HHTTH|θ) = P(H|θ)P(H|θ)P(T|θ)P(T|θ)P(H|θ)
P(HHTTH|θ) = θθ(1−θ)(1−θ)θ
P(HHTTH|θ) = θ<sup>3</sup>(1−θ)<sup>2</sup>

# Transforming a problem
• Suppose that we want to maximise  f
• i.e.  x* = argmaxf(x)
                x∈𝒳

• If we have another function, g, is monotonically 
increasing, then 
• i.e. x* = argmaxf(x) = argmaxg(f(x))
          x∈𝒳                 x∈𝒳

# Log-liklihood
• Log is monotonically increasing 
• Hence,  x* = argmaxf(x) = argmaxlog(f(x))
                x∈𝒳            x∈𝒳
• Hence, by taking log, we can transform the likelihood 
replace the product with a sum! 
• We call this the log-likelihood

# Taking Natural Log In png file

Hence 
argmaxθ3(1 − θ2) = argmaxlog(θ3(1 − θ2))
θ∈ℝ                   θ∈ℝ
argmaxθ3(1 − θ2) = argmaxlog((θ3) + log(1 − θ)2)
θ∈ℝ                  θ∈ℝ
argmaxθ3(1 − θ2) = argmax3log(θ) + 2log(1 − θ)
θ∈ℝ                  θ∈ℝ


# MLE using Gradient Descent
• Some distributions don’t have closed form solutions for 
Log-liklihood 
• Or solving closed forms are difficult 
• We can use gradient descent to solve such cases! 
• Need to transform maximisation problem to minimisation 
problem

# Negative log-likelihood
• Recall, by multiplying by -1, we can transform a 
maximisation problem to a minimisation problem 
• Hence, by negating the log-likelihood, we get a loss 
function we can use with gradient descent

# General NLLs
• The loss function need not be tied to a specific dataset 
• Can derive NLLs to use as loss functions that are 
expressed in terms of a generalised dataset






