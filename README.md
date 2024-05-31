# Reproducibility code for AggInc: Efficient Aggregated Kernel Tests using Incomplete U-statistics

This GitHub repository contains the code for the reproducibility of the experiments in our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf).

To use our MMDAggInc, HSICAggInc and KSDAggInc tests in practice, we recommend using our `agginc` package, more details available on the [agginc](https://github.com/antoninschrab/agginc) repository.

The code for reproducibility of the experiments of our paper, and for generating the figures in [figures](figures), is presented in the notebook [experiments.ipynb](experiments.ipynb). The outputs of all the experiments are saved in [results](results).

## Requirements
- `python 3.9`

The packages in [requirements.txt](requirements.txt) are required to run our tests and the ones we compare against. 
The `numpy` package version `<= 1.21` is only needed for compatibility with the `theano` package.

Additionally, the `jax` and `jaxlib` packages are required to run the Jax implementation of AggInc in [agginc/jax.py](agginc/jax.py).

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```
git clone git@github.com:antoninschrab/agginc-paper.git
cd agginc-paper
```
We then recommend creating and activating a virtual environment by either 
- using `venv`:
  ```
  python3 -m venv agginc-env
  source agginc-env/bin/activate
  # can be deactivated by running:
  # deactivate
  ```
- or using `conda`:
  ```
  conda create --name agginc-env python=3.9
  conda activate agginc-env
  # can be deactivated by running:
  # conda deactivate
  ```
The required packages can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
```

For using the Jax implementation of our tests, Jax needs to be installed, for which we recommend using `conda`. This can be done by running
- for GPU:
  ```bash
  conda install -c conda-forge -c nvidia pip cuda-nvcc "jaxlib=0.4.1=*cuda*" jax
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip jaxlib=0.4.1 jax
  ```

## How to use MMDAggInc, HSICAggInc and KSDAggInc in practice?

The MMDAggInc, HSICAggInc and KSDAggInc tests are implemented as the function `agginc` in [agginc/np.py](agginc/np.py) for the Numpy version and in [agginc/jax.py](agginc/jax.py) for the Jax version.

For the Numpy implementation of our AggInc tests, we only require the `numpy`, `scipy` and `psutil` packages.

For the Jax implementation of our AggInc tests, we only require the `jax`, `jaxlib` and `psutil` packages.

To use our tests in practice, we recommend using our `agginc` package which is available on the [agginc](https://github.com/antoninschrab/agginc) repository. It can be installed by running
```bash
pip install git+https://github.com/antoninschrab/agginc.git
```
Installation instructions and example code are available on the [agginc](https://github.com/antoninschrab/agginc) repository. 

We also provide some code showing how to use our AggInc tests in [demo.ipynb](demo.ipynb). 

In practice, we recommend using the Jax implementation as it runs considerably faster.

## Speed comparison

We recommend using our Jax implementation in [agginc/jax.py](agginc/jax.py) over our Numpy implementation in [agginc/np.py](agginc/np.py) as it runs more than 100 times faster after compilation, as can be seen from the results in the notebook [speed.ipynb](speed.ipynb) which are reported below.

| Speed in ms | Numpy (CPU) | Jax (CPU) | Jax (GPU) | 
| -- | -- | -- | -- |
| MMDAggInc | 4490 | 844 | 23 | 
| HSICAggInc | 2820 | 539 | 18 |
| KSDAggInc | 3770 | 590 | 22 | 

## References

In our experiments, we compare MMDAggInc to
- Tests: ME (Mean Embedding) and SCF (Smooth Characteristic Function)
- Paper: [Interpretable Distribution Features with Maximum Testing Power](https://proceedings.neurips.cc/paper/2016/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Kacper Chwialkowski, Arthur Gretton
- Code: [interpretable-test
 repository](https://github.com/wittawatj/interpretable-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

and to

- Test: PSI OST (Post Selection Inference One-Sided Test)
- Paper: [Learning Kernel Tests Without Data Splitting](https://proceedings.neurips.cc/paper/2020/file/44f683a84163b3523afe57c2e008bc8c-Paper.pdf)
- Authors: Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, Krikamol Muandet
- Code: [tests-wo-splitting repository](https://github.com/MPI-IS/tests-wo-splitting) by [Max Planck Institute for Intelligent Systems](https://github.com/MPI-IS)

We compare HSICAggInc to 
- Test: FSIC (Finite Set Independence Criterion)
- Paper: [An Adaptive Test of Independence with Analytic Kernel Embeddings](http://proceedings.mlr.press/v70/jitkrittum17a/jitkrittum17a.pdf)
- Authors: Wittawat Jitkrittum, Zoltán Szabó, Arthur Gretton
- Code: [fsic-test
 repository](https://github.com/wittawatj/fsic-test) by [Wittawat Jitkrittum](https://github.com/wittawatj)

We compare KSDAggInc to
- Test: FSSD (Finite Set Stein Discrepancy)
- Paper: [A Linear-Time Kernel Goodness-of-Fit Test](https://papers.nips.cc/paper/2017/file/979d472a84804b9f647bc185a877a8b5-Paper.pdf)
- Authors: Wittawat Jitkrittum, Wenkai Xu, Zoltán Szabó, Kenji Fukumizu, Arthur Gretton
- Code: [kernel-gof repository](https://github.com/wittawatj/kernel-gof) by [Wittawat Jitkrittum](https://github.com/wittawatj)

and to

- Test: LSD (Learning Stein Discrepancy)
- Paper: [Learning the Stein Discrepancy
for Training and Evaluating Energy-Based Models without Sampling](http://proceedings.mlr.press/v119/grathwohl20a/grathwohl20a.pdf)
- Authors: Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Richard Zemel
- Code: [LSD repository](https://github.com/wgrathwohl/LSD) by [Will Grathwohl](https://github.com/wgrathwohl)

and to

- Tests: L1 IMQ and Cauchy RFF (Random Fourier Feature)
- Paper: [Random Feature Stein Discrepancies](https://proceedings.neurips.cc/paper/2018/file/0f840be9b8db4d3fbd5ba2ce59211f55-Paper.pdf)
- Authors: Jonathan H. Huggins, Lester Mackey
- Code: [RFSD repository](https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/) by [Jonathan Huggins](https://bitbucket.org/jhhuggins/)

## Contact

If you have any issues running our code, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@inproceedings{schrab2022efficient,
  author    = {Antonin Schrab and Ilmun Kim and Benjamin Guedj and Arthur Gretton},
  title     = {Efficient Aggregated Kernel Tests using Incomplete {$U$}-statistics},
  booktitle = {Advances in Neural Information Processing Systems 35: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022},
  editor    = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year      = {2022},
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test 
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dpkernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
- [dckernel](https://github.com/antoninschrab/dckernel/): Robust to Data Corruption dcMMD dcHSIC tests
