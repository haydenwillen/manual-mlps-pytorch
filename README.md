# Manual MLPs in PyTorch for Classification and Regression

**Hayden Willen** | DS 4021 Machine Learning II | University of Virginia School of Data Science

This project builds simple neural networks by hand in PyTorch for three tasks, which are binary classification, multi-class classification, and regression. Each model sets up its own weights, makes its own predictions, measures its own error, and updates its own weights, without using PyTorch's built-in layers or optimizers.

Building each part by hand makes it easier to understand how neural networks learn and why they sometimes fail.

## What's in the notebook

1. **Design Choices.** The cost function and output layer for each task, why we standardize the data, and how we check for overfitting.
2. **Binary Classification.** A model that sorts points into two curved groups from `make_moons`.
3. **Multi-Class Classification.** A model that sorts points into three groups from `make_classification`, where some labels are wrong on purpose.
4. **Regression.** A model that predicts a number from noisy data made with `make_regression`.
5. **Code Reuse with Inheritance.** How one shared parent class could cut down the repeated code in the three models.

Each model was trained 16 ways. We tried ReLU and tanh, one and two hidden layers, and 2, 8, 32, and 128 hidden units. Every model used an 80/20 train and holdout split, a learning rate of 0.01, 200 epochs, and a batch size of 32.

## Results

This table shows the best holdout score for each activation and number of layers. The full tables for all 16 settings are in the notebook.

| Activation | Hidden layers | Binary accuracy | Multi-class accuracy | Regression R² |
|---|---|---|---|---|
| ReLU | 1 | 95.0% | 85.0% | **0.443** |
| ReLU | 2 | 97.5% | 84.5% | 0.439 |
| tanh | 1 | 97.0% | 84.5% | 0.440 |
| tanh | 2 | **99.0%** | **85.5%** | 0.439 |

## Key findings

- **Bigger models helped on curved data.** On the moons data, holdout accuracy went from about 92% with 2 hidden units to 99% with two tanh layers of 128 units. The two groups are curved, so the model needs more units to draw a curved line between them.
- **Noise limited the other two tasks.** In the multi-class data, about 15% of the labels are wrong, and every model scored close to 85%. In the regression data, the added noise alone causes most of the error. Bigger models could not get past these limits.
- **ReLU broke in the smallest model.** Two ReLU layers with only 2 units scored exactly 50% on the moons data, which means it guessed the same class for every point. Both units got stuck at zero and stopped learning. Tanh did not have this problem and scored 91.5% with the same size.
- **Only the regression models overfit.** They did better on the training set than the holdout set, and the gap got bigger as the models got bigger. In both classification tasks, the models actually did better on the holdout set.
- **Each activation had a small edge on different data.** Tanh did a little better on the curved classification data, and ReLU did a little better on the straight-line regression data.

## Data

All three datasets are made in the notebook with scikit-learn, so there are no data files.

- **Binary.** `make_moons(n_samples=1000, noise=0.2, random_state=42)`
- **Multi-class.** `make_classification(n_samples=1000, n_classes=3, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.15, random_state=1234)`
- **Regression.** `make_regression(n_samples=1000, n_features=2, n_informative=2, noise=75, random_state=1234)`

## How to run

```bash
git clone https://github.com/haydenwillen/manual-mlps-pytorch.git
cd manual-mlps-pytorch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook manual_mlps_pytorch.ipynb
```

## Project structure

```
manual-mlps-pytorch/
├── manual_mlps_pytorch.ipynb   # the three models, results and discussion
├── requirements.txt            # packages needed to run the notebook
├── .gitignore                  # files Git should ignore
└── README.md                   # this file
```

## Limitations

- **One random seed.** Each result comes from one run. The holdout set has 200 points, so a difference of 1 or 2 percent could just be chance.
- **No tuning.** Every model used the same learning rate, epochs, and batch size. Some models might do better with different settings.
- **Holdout scores beat training scores in classification.** This is likely because the holdout set happened to have fewer wrong labels, not because the models are better on new data.
- **The regression target was standardized.** Its values were in the hundreds, which can destabilize training. We changed the predictions back to the original units before measuring the error.
- **Simple data.** Each dataset has only 2 features, so these results may not hold for real data with many features.

## Credits

This started as a group lab for DS 4021 Machine Learning II at the University of Virginia. I wrote the binary and multi-class classification sections, and my groupmates wrote the others. For this version, I reworked everyone's code into one notebook, reviewed the full notebook, edited the findings, and completed the remaining analysis.
