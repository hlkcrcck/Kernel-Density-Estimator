#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from kde import ParallelVectorKernelDensityEstimator
from utils import read_mnist_data
import argparse
import timeit

parser = argparse.ArgumentParser(description='Benchmark MNIST on different algorithms and data samples')
parser.add_argument("mnist_path", help="MNIST pickle path")
args = parser.parse_args()

if __name__ == '__main__':
    print("Reading data...")
    X_A, X_B = read_mnist_data(args.mnist_path)

    print("Calculating the score...")
    start_time = timeit.default_timer()
    mean_log_proba = ParallelVectorKernelDensityEstimator(X_A, standard_deviation=0.2) \
        .score(X_B)
    elapsed = timeit.default_timer() - start_time
    print("Mean log probability is: {}, Time Elapsed: {}".format(mean_log_proba, elapsed))

