# Machine Learning

#The phenomena of human intelligence
• One of the most intriguing and defining characteristics of 
human intelligence is our ability to learn! 
• If our goal is to build AI tools that are useful to us, it 
would be useful to build agents capable of learning 
• What is learning?

# What is Learning?
• Broadly speaking, we use experience to construct mental 
models to encode understanding 
• Use mental models to reason about situations that we have 
not yet experienced 
• Introspect mental models to gain insight into what we 
experienced 
• Use mental models to help make decisions about which 
actions to taken 
• Is different from memorising data


# What is Machine Learning?
• The usage of algorithms to computationally and in an 
automated fashion derive models from data such that we 
can glean non-trivial insight, make predictions about 
future events and unseen scenarios, and make decisions 
• Recall our definition of a rational agent 
• Uses knowledge of state 
• To make decisions 
• To optimise performance measure

# Types of Machine 
Learning
• Learning is broad. Likewise, there 
are many flavours of machine 
learning with different use cases! 
• Three major categories: 
• Supervised Learning 
• Unsupervised Learning 
• Reinforcement Learning 
• Others exist, but these are 
typically grey areas around these 
main three, e.g. weakly 
supervised learning, semi
supervised learning, etc.

# Types of Machine Learning
• Will focus on some supervised learning 
• In the space of unsupervised learning, will only be able to 
cover autoencoders 
• Won’t have time to cover reinforcement learning at all 
• But should know definitions and sub-types of problems 
• Can’t search for a solution unless you know the 
problem you are trying to solve!

# On Data and Features
• Most machine learning algorithms assume that your data is numerically 
encoded in some way 
• For the most part, we store our encodings for some data as a tensor of some 
sort 
• We will work with vectors and matrices here 
• Each component of tensor represents some aspect of characteristic of our data 
• Each characteristic is termed a feature 
• Example, we can measure the width and length of sepals and petals of 
f
lowers. Each measurement has a numeric value, that we can then 
encapsulate in a consistent way as feature vectors

# Types of data
• Different types of data: 
• Quantitative: regular numerical data that can be real numbers, e.g. 
height and weight 
• Ordinal: integer data where ordering of numbers is important, e.g. 
preference scores in surveys 
• Categorical: label data (usually represented as an integer) where 
order does not matter and numerical association is usually 
“arbitrary” and to the user’s discretion, e.g. dog (0) or cat (1) 
• If we have more than 2 categories, we can either label each an 
arbitrary integer or use one-hot encoding

# One Hot Encoding
• Suppose that we have n categories. We let every 
category have an associated integer between 0 and 
n −1 and represent vector as n-length bit vector with all 
zeros except in the one component associated with the 
category. 

• Example, suppose we have three colours of flowers: 
pink, purple, and yellow. Let pink = 0, purple = 1, yellow 
= 2. Our flower is purple, so we encode this as  [0,1,0]<sup>T</sup>
and we append to the rest of the feature vector!  

# Supervised Learning
• Given a set of input-output pairs.
𝒟={(x1,y1),(x2,y2),…,(x1,y1)} where 
∀i, xi ∈ 𝒳 and  ∀i, yi ∈ Y

• Assume that there exists some function f : 𝒳 →𝒴 f
can be some “process” or relationship, i.e. the 
relationship between height, weight, average weekly 
caloric intake, and Haemoglobin A1C (HbA1C) 

• We want to approximate f 

•x1, x2, …,xn are our independent or manipulated variables
•y1, y2, …,yn are our dependent or responding variables 

• Want to use independent variables’ values to predict dependent 
variable values! 
• In essence, supervised learning is about trying to draw best-fit lines

• Two main subtypes: 
• Regression: dependent variable is continuous or discrete and infinite 
(e.g. integers). E.g. predicting house prices from square footage, 
number of bedrooms, number of bathrooms, etc.. 
• Classification: dependent variable is discrete and finite. We also 
assume that the dependent variable is categorical. E.g. building a spam 
classifier, building a tool to recognise emotions in facial expressions 
• Ordinal Regression: can be used when the dependent variable is 
ordinal, E.g. determining the if some someone will strongly disagree, 
disagree, agree, or strongly agree with an opinion given 
demographic information

# Hypothesis Space in Supervised Learning
• Recall that we want to find a function that approximates 
the relationship between independent and dependent 
variables well 
• But space of possible functions is infinite and very varied  
• Need to make simplifying assumptions of what sort of 
functions would work well enough 
• Usually, we make assumptions about general structure of 
functions, e.g. linear, polynomial, etc…

# Hypothesis Space in Machine Learning
• For example, we say that we are only considering linear functions. 
Hence, our hypothesis space is set of all linear functions from our 
input space (ℝn) to output space (ℝ). 

• Every linear function over some ℝn is of the form wTx +c. Here x is our function’s input. w and c are parameters of our linear function. We can formalise this by saying f(x; w, c) = wTx + c

• Need to make decisions about values for w and c

• Hence, we reduce the problem of finding a good linear function to 
approximate our relationship to finding good values of w and c

# Empirical Risk Minimisation (ERM)
• How to choose good parameter values? 
• Assume that our data is a representative sample 
• Need to be able to measure how badly we are 
performing on average across data points as proxy for 
how badly we will perform in the wild 
• This function that computes this acts as a loss function 
that we want to minimise!

# Empirical Risk Minimisation - Important to use umage to understand


# Underfitting and Overfitting
• So we used a technique to find a good set of parameters 
according to our hypothesis space 
• How do we know our hypothesis space was suitable or 
not? 
• Our model needs to be able to learn the relationships 
• But don’t want our model to memorise relationships (think 
memorising selection sort without understanding how it 
works)

# Variance-Bias Tradeoff
• Underfitting and overfitting are related to two statistical properties of 
our model - the model’s variance and the model’s bias 
• Won’t belabour formalism here, but important to get intuition of 
these concepts 
• Think of variance as representing how “sensitive” the model is to 
perturbations in the input 
• Think of bias as representing how consistent the model is in learning 
incorrect concepts 
• A model with fewer parameters have less variance but more bias 
and vice versa

# Unsupervised Learning
• Given data points without dependent variables or labels 
• Want to gain insight into data or preprocess for other 
machine learning algorithms 
• Sometimes also called knowledge discovery or pattern 
mining 
• Several types of unsupervised learning


# Clustering
• Discover groups within data 
with similar characteristics 
• Used in fields such as 
marketing, sociology, and 
biology 
• Graph specific variant called 
community detection
• Techniques: K-Means, Louvain 
Modularity, DBSCAN, 
HDBSCAN

# Anomaly Detection
• Detect novel or unexpected 
events or data points
• Used in medical data analysis 
and tasks such as credit card 
fraud detection and intrusion 
detection
• Can also be used in any sort of 
signal processing task
• Sometimes uses clustering as 
preprocessing step
• Examples: clustering-based, 
local outlier factor

# Dimensionality Reduction and Feature Learning
• Given some data, compress data 
into lower dimensional space
• Or learn projection of data into 
another space (called the latent 
feature space)
• Useful for data visualisation 
• Or as preprocessing step
• Examples: PCA, TSNE, UMAP, 
Word2Vec, BERT, Node2Vec, 
StEM, Autoencoders
• Many use matrix factorisations 
or self-supervision

# Association Rule Mining
• Find common associations 
between items
• Used in fields such as 
recommendation systems, 
market basket analysis, and 
insurance 
• Examples: APriori Algorithm

# Density Estimation
• Given data, try to fit data non
parametrically
• Can be used as preprocessing 
step to anomaly detection or to 
some supervised learning 
techniques 

# Reinforcement Learning 
• Agents can interact with environment 
• Receives state s<sub>t</sub>
• Executes action a<sub>t</sub>
• Transitions to state s<sub>t+1</sub> based on a<sub>t</sub> and earns reward r<sub>t</sub>
• Such as circumstance can be described by something called a Markov Decision 
Process (MDP) 
• We want to come up with policy function π such that we earn the most rewards at 
the end 
• Examples of use cases: Playing Atari Games, teaching robots how to walk, and 
protein folding


# Markov Decision Process Is under the image of "Markov_Decision_Principle"

# Reinforcement Learning

• Reinforcement Learning tries to learn policy function π : S →A that maximises reward at the end 
• Examples of techniques: Q-Learning, Deep Q-Learning 
• Interestingly enough, the genetic algorithm and related 
techniques have been show to outperform gradient 
based optimisation for learning π

# Machine Learning Experiments
• Important to adopt proper, scientifically robust 
methodology to conduct machine learning 
• Good experimental design is important 
• Can’t always prove things formally :’(  
• So, good experiments is sometimes all that we have 
• Interesting research done at level of methodology!

# Training Set and Testing Set
• Remember that our models are supposed to work for a 
population of scenarios 
• But only have samples (we hope are representative) 
• How can we conclude that our supervised learning 
models work well?

# Model Training Diagram Is in the images, this is important

# Evaluating Performance
• Need to use appropriate way to measure model 
performance on testing set 
• The appropriate metric depends on the problem type and 
characteristics  
• Note the metrics and the loss function used are not 
necessarily the same!