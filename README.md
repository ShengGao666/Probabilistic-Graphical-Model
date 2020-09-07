# Probabilistic-Graphical-Model
* ## A1 regression&nueral net
The goal of this assignment is to get you familiar with the basics of decision theory and
gradient-based model fitting(regression&nueral net).
* ## A2 Stochastic Variational Inference in the TrueSkill Model
The goal of this assignment is to get you familiar with the basics of Bayesian inference in large models
with continuous latent variables, and the basics of stochastic variational inference.
Background We'll implement a variant of the TrueSkill model, a player ranking system for competitive
games originally developed for Halo 2. It is a generalization of the Elo rating system in Chess. For the
curious, the original 2007 NIPS paper introducing the trueskill paper can be found here: http://papers.
nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system.pdf
This assignment is based on one developed by Carl Rasmussen at Cambridge for his course on probabilistic
machine learning: http://mlg.eng.cam.ac.uk/teaching/4f13/1920/
* ## A3 Variational Autoencoders
In this assignment, we will implement and investigate the Variational Autoencoder on binarized MNIST
digits, as introduced by the paper Auto-Encoding Variational Bayes by Kingma and Welling (2013). Before
starting, we recommend reading this paper.
Data. Each datapoint in the MNIST dataset is a 28x28 grayscale image (i.e. pixels are values between 0
and 1) of a handwritten digit in f0 : : : 9g, and a label indicating which number. MNIST is the `fruit 
y' of
machine learning { a simple standard problem useful for comparing the properties of different algorithms.
Use the rst 10000 samples for training, and the second 10000 for testing. Hint: Also build a dataset of
only 100 training samples to use when debugging, to make loading and training faster.
