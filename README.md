# Train excitatory-inhibitory recurrent neural networks for cognitive tasks

## Forked: Training RNNs for discriminatory tasks

The added tasks workingMemory.py, romoOne.py and its generalizations are working. They are currently being analyzed to see how we can improve training speed and understand the dynamics of the trained networks.


## Requirements

This code is written in Python 2.7 and requires

* [Theano 0.7](http://deeplearning.net/software/theano/)

Optional but recommended if you plan to run many trials with the trained networks outside of Theano:

* [Cython](http://cython.org/)

Optional but recommended for analysis and visualization of the networks (including examples from the paper):

* matplotlib

The code uses (but doesn't require) one function from the [NetworkX](https://networkx.github.io/) package to check if the recurrent weight matrix is connected (every unit is reachable by every other unit), which is useful if you plan to train very sparse connection matrices.

## Installation

Because you will eventually want to modify the `pycog` source files, we recommend that you "install" by simply adding the `pycog` directory to your `$PYTHONPATH`, and building the Cython extension to (slightly) speed up Euler integration for testing the networks by typing

```
python setup.py build_ext --inplace
```

You can also perform a "standard" installation by going to the `pycog` directory and typing

```
python setup.py install
```

## Examples


Example task specifications, including those used to generate the figures in the paper, can be found in `examples/models`.

Training and testing networks involves some boring logistics, especially regarding file paths. You may find the script `examples/do.py` helpful as you start working with your own networks. For instance, to train a new network we can just type (from the `examples` directory)

```
python do.py models/sinewave train
```

For this particular example we've also directly included code for training and plotting the result, so you can simply type

```
python models/sinewave.py
```



## Notes

C.F. The original source https://github.com/frsong/pycog/blob/master/README.md

## License

MIT

## Citation

This code is the product of work carried out in the group of [Xiao-Jing Wang at New York University](http://www.cns.nyu.edu/wanglab/). If you find our code helpful to your work, consider giving us a shout-out in your publications:

* Song, H. F.\*, Yang, G. R.\*, & Wang, X.-J. "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework." *PLoS Comp. Bio.* 12, e1004792 (2016). (\* = equal contribution)
