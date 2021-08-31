## Usage

```bash
python main.py <MNIST_PATH>
```
*http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

## Run Benchmarks
You can provide multiple sizes and algorithms. 


```bash
python benchmarks.py <MNIST_PATH> -sizes [10 ...] -algos [pseudo ...]
```

Available algorithms : pseudo, vector, vector_parallel, scikit*

To use scikit in the benchmark install it first:
```bash
pip install sklearn
```

## Benchmarks
| Data Size/Algorithm |   10   |   100  |  1000 |   10000  |
|:-------------------:|:------:|:------:|:-----:|:--------:|
|      PseudoKDE      |  0.15s | 20.70s |   -   |     -    |
|      VectorKDE      | 0.001s |  0.01s | 0.91s | 163.09s |
|  ParallelVectorKDE  |  0.62s |  0.65s | 1.20s |  78.72s |
|    Scikit-Learn*    | 0.001s |  0.01s | 0.94s |  109.75s |

*https://scikit-learn.org/stable/