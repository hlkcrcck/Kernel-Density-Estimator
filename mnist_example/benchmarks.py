#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from kde import KernelDensityEstimator
from kde import VectorKernelDensityEstimator
from kde import ParallelVectorKernelDensityEstimator

import timeit
from utils import read_mnist_data
import argparse

parser = argparse.ArgumentParser(description='Benchmark MNIST on different algorithms and data samples')
parser.add_argument("mnist_path", help="MNIST pickle path")
parser.add_argument('-sizes', type=int, nargs='+',
                    help='benchmark different sizes')
parser.add_argument('-algos', type=str, nargs='+',
                    help='benchmark different algorithms')
args = parser.parse_args()

if __name__ == '__main__':
    algorithms = {"pseudo": KernelDensityEstimator,
                  "vector": VectorKernelDensityEstimator,
                  "vector_parallel": ParallelVectorKernelDensityEstimator}

    for size in args.sizes:
        X_A, X_B = read_mnist_data(args.mnist_path, size)

        for algo in args.algos:
            start_time = timeit.default_timer()
            if algo == 'sklearn':
                from sklearn.neighbors import KernelDensity
                # instantiate and fit the KDE model
                kde = KernelDensity(bandwidth=0.2, kernel='gaussian', leaf_size=10000)
                kde.fit(X_A)
                # score_samples returns the log of the probability density
                log_prob = kde.score_samples(X_B)
                mean_log_likelihood = sum([e for e in log_prob]) / len(log_prob)
            else:
                mean_log_likelihood = algorithms[algo](X_A, standard_deviation=0.2) \
                    .score(X_B)

            elapsed = timeit.default_timer() - start_time
            print("Algo: {}, Data Size: {}, MLL: {}, Time: {}, "
                  .format(algo, size, mean_log_likelihood, elapsed))
