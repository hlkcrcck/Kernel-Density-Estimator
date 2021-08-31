#Kernel Density Estimator
[Benchmarks](./mnist_example/README.md#Benchmarks)

## Installation

Run the following to install this package:

```python
pip install .
```

For development purposes:
```python
pip install -e .[dev]
```

## To run tests:
```python
pytest
```

## Usage:
```python
from kde import ParallelVectorKernelDensityEstimator as KDE

kde = KDE(X_A, standard_deviation=0.2)

# to observe probabilities
log_proba = kde.score_samples(X_B)

# mean log probability
mean_log_proba = kde.score(X_B)


```

## Examples

[MNIST](./mnist_example/README.md)