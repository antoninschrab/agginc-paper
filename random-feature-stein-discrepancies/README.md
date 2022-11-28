The `rfsd` package was used produce the experiments for:

[Jonathan H. Huggins](http://www.jhhuggins.org),
[Lester Mackey](https://web.stanford.edu/~lmackey/).
*[Random Feature Stein Discrepancies](https://arxiv.org/abs/1806.07788)*.
In *Proc. of the 32nd Annual Conference on Neural Information Processing
Systems* (NIPS), 2018.

# Installation

To `rfsd` package requires you to install the `kgof` package:
```bash
pip install git+https://github.com/wittawatj/kernel-gof.git
```
Then:
```bash
pip install git+https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies.git
```
The experiments from the paper can be reproduced by running each
[script](scripts/) with its default arguments. Or, in the
case of [run_gof_experiment.py](scripts/run_gof_experiment.py), once
with each of the commmands `null`, `laplace`, `student-t`, and `rbm`.
