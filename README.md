Neural Network Training with a Genetic Algorithm
Lab assignment for the Artificial Intelligence course at FER, June 2026.
Implemented a feedforward neural network in Python using NumPy and trained it with a genetic algorithm. 
Supports three network architectures and reports training and test mean squared error.

Install NumPy with pip install numpy, then run:
python solution.py --train train.csv --test test.csv --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000

Both datasets should be CSV files with a header, numeric values, and the target in the last column.
