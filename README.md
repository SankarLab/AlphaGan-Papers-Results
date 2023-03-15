
# Towards Addressing GAN Training Instabilities: Dual-Objective GANs with Tunable Parameters

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

In an effort to address the training instabilities of GANs, we introduce a class of dual-objective GANs with different value functions (objectives) for the generator (G) and discriminator (D). In particular, we model each objective using  α-loss, a tunable classification loss, to obtain (α_D, α_G)-GANs, parameterized by (α_D, α_G) ∈ [0, ∞)^2. For sufficiently large number of samples and capacities for G and D, we show that the resulting non-zero sum game simplifies to minimizing an f-divergence under  appropriate conditions on (α_D, α_G). In the finite sample and capacity setting, we define estimation error to quantify the gap in the generator's performance relative to the optimal setting with infinite samples and obtain upper bounds on this error, showing it to be order optimal under certain conditions. Finally, we highlight the value of tuning (α_D, α_G) in alleviating training instabilities for the synthetic 2D Gaussian mixture ring and the Stacked MNIST datasets.


## Authors

- Monica Welfert ([@mwelfert](https://www.github.com/mwelfert))
- Kyle Otstot ([@kotstot6](https://www.github.com/kotstot6))
- Gowtham R. Kurri ([@gowthamkurri](https://www.github.com/gowthamkurri))
- Lalitha Sankar ([@lalithaSankarASU](https://www.github.com/lalithaSankarASU))


## Technologies

**Backend:** Python, PyTorch, NumPy

**Cluster Environment:** Slurm, SBATCH


## Experiment Results

#### Dataset: 2D-Ring `python3 experiment1.py --dataset 2Dring`

| Objective | Type | Parameters | Success | Failure |
| :--------: | :-------: | :--------: | :--------: | :--------: |
| Cross Entropy | S | | 0.5 | 0.5 |
| (a_D, a_G) GAN | S | α_D = 0.3, α_G = 1 | **0.5** | 0.5 |
| Cross Entropy | NS | | 0.5 | 0.5 |
| (a_D, a_G) GAN | NS | α_D = 0.5, α_G = 1.2 | 0.5 | **0.5** |
| LSGAN | NS | a = 0, b = 1, c = 0 | 0.5 | 0.5 |

#### Dataset: Stacked MNIST `python3 experiment2.py`

| Objective | Type | Parameters | Modes | FID |
| :--------: | :-------: | :--------: | :--------: | :--------: |
| Cross Entropy | NS | | 100 ± 100 | 100 ± 100 |
| (a_D, a_G) GAN | NS | α_D = 2, α_G = 2 | **100 ± 100** | **100 ± 100** |
| LSGAN | NS | a = 0, b = 1, c = 0 | 100 ± 100 | 100 ± 100 |

## Run Experiment

Clone the project

```bash
  git clone https://github.com/SankarLab/AlphaGan-Papers-Results.git
```

Go to the project directory

```bash
  cd /path/to/AlphaGan-Papers-Results
```

#### Run Experiment 1 (2D-Ring):

```bash
  python3 experiment1.py ( ... parameters ...)
```

|Parameter|Type|Default|Description|
| :--- | :--- | :--- | :--- |
|`--seed`|`int`|`1`|random seed for reproducibility|
|`--dataset`|`str`|`'2Dring'`|dataset type|
|`--train_size`|`int`|`50000`|number of train examples|
|`--test_size`|`int`|`25000`|number of test examples|
|`--batch_size`|`int`|`128`|batch size used during training/testing|
|`--save_bursts`| |`False`|saves the plotted output bursts for each checkpoint|
|`--d_layers`|`int`|`2`|number of hidden layers in discriminator|
|`--g_layers`|`int`|`2`|number of hidden layers in generator|
|`--d_width`|`int`|`200`|hidden layer width in discriminator|
|`--g_width`|`int`|`400`|hidden layer width in generator|
|`--n_epochs`|`int`|`400`|number of epochs for training|
|`--epoch_step`|`int`|`401`|number of epochs between validation checkpoints|
|`--d_lr`|`float`|`0.0001`|learning rate for discriminator|
|`--g_lr`|`float`|`0.0001`|learning rate for generator|
|`--d_alpha`|`float`|`1.0`|alpha parameter for discriminator|
|`--g_alpha`|`float`|`1.0`|alpha parameter for generator|
|`--non_saturating`| |`False`|uses non saturating loss function|
|`--ls_gan`| |`False`|uses LS GAN loss functions|

#### Run Experiment 2 (Stacked MNIST):

```bash
  python3 experiment2.py ( ... parameters ...)
```

|Parameter|Type|Default|Description|
| :--- | :--- | :--- | :--- |
|`--seed`|`int`|`1`|random seed for reproducibility|
|`--train_size`|`int`|`100000`|number of train examples|
|`--test_size`|`int`|`25000`|number of test examples|
|`--noise_dim`|`int`|`100`|dimensionality of latent noise vectors|
|`--batch_size`|`int`|`64`|batch size used during training/testing|
|`--save_bursts`| |`False`| saves the plotted output bursts for each checkpoint|
|`--n_epochs`|`int`|`50`|number of epochs for training|
|`--epoch_step`|`int`|`51`|number of epochs between validation checkpoints|
|`--d_lr`|`float`|`0.001`|learning rate for discriminator|
|`--g_lr`|`float`|`0.001`|learning rate for generator|
|`--d_width`|`int`|`1`|channel multiplier for discriminator|
|`--g_width`|`int`|`1`|channel multiplier for generator|
|`--beta1`|`float`|`0.9`|beta1 parameter for adam optimization|
|`--d_alpha`|`float`|`1.0`|alpha parameter for discriminator|
|`--g_alpha`|`float`|`1.0`|alpha parameter for generator|
|`--non_saturating`| | `False`|uses non saturating loss function|
|`--ls_gan`| | `False`|uses LS GAN loss functions|

#### Slurm Environment

In `run_experiment<X>.py`, adjust `grid` dictionary used for hyperparameter grid search. Example:

```python
grid = {
    'seed' : list(range(1,101)),
    'non_saturating' : [True],
    'g_lr' : [1e-4, 2e-4, 5e-4, 1e-3],
}
```

Adjust the `run_experiment<X>.sh` SBATCH script for use.

Lastly, run the following command:

```bash
python3 run_experiment<X>.py --sbatch
```
