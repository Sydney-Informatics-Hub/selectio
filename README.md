# Selectio: Multi-Model Feature Importance Scoring and Auto Feature Selection.

This Python package provides computation of multiple feature importance scores, feature ranks,
and automatically suggests a feature selection based on the majority vote of all models.

## Models

Currently the following six models for feature importance scoring are included:
- Spearman rank analysis (see 'selectio.models.spearman')
- Correlation coefficient significance of linear/log-scaled Bayesian Linear Regression (see 'selectio.models.blr')
- Random Forest Permutation test (see 'selectio.models.rf')
- Random Decision Trees on various subsamples of data (see 'selectio.models.rdt')
- Mutual Information Regression (see 'selectio.models.mi')
- General correlation coefficients (see 'selectio.models.xicor')

Moreover, this package includes multiple functions for visualisation of feature ranking and hierarchical feature clustering.

Note that the current feature importance models support numerical data only. Categorical data 
will need to be encoded to numerical features beforehand.

## Installation

```bash
pip install selectio
```

or for development as conda environment:

```bash
conda env update --file environment.yaml
conda activate selectio
```

## Requirements

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- pyyaml

See file environment.yaml for more details.

## Usage

There are multiple options to compute feature selection scores 

Option 1) computed directly using the class Fsel, e.g.

```python
from selectio.selectio import Fsel
# Read in data X (nsample, nfeatures) and y (nsample)
fsel = Fsel(X, y)
# Score features and return results as dataframe:
dfres = fsel.score_models()
```
This returns a table with all scores and feature selections. 

Option 2) with a settings yaml file (template provided) that includes more functionality (including preprocessing and plotting), e.g:
```python
from selectio import selectio
# Read in data from file, generate feature importance plots and save results as csv:
selectio.main('settings_featureimportance.yaml')
```
This will automatically save all scores and selections in csv file and create multiple score plots.


Option 3) if installed locally as standalone script with a settings file:
```bash
cd selectio
python selectio.py -s <FILENAME>.yaml
```

User settings such as input/output paths and all other options are set in the settings file 
(Default filename: settings_featureimportance.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python selectio.py -s settings/settings_featureimportance.yaml).

## Settings YAML file

For settings file template, see [here](https://github.com/sebhaan/selectio/blob/main/selectio/settings/settings_featureimportance.yaml)

The main settings are:
```yaml
# Input data path
inpath: ...

# File name with soil data and corresponding covariates
infname: ...

# Output results path
outpath: ...

# Name of target for prediction (column name in dataframe)
name_target: ...

# Name or List of features (column names in infname)
# (covariates to be considered )
name_features: 
- ...
- ...
```


## Simulation and Testing

The selectio package provides the option to generate simulated data (see `selectio.simdata`) 
and includes multiple test functions (see `selectio.tests`), e.g.

```python
from selectio import tests
tests.test_select()
```

For more examples and how to create simulated  via `simdata.py`, see the provided Jupyter notebooks `feature_selection.ipynb`.


## Adding Custom Model Extensions

More models for feature scoring can be added in the folder 'models' following the existing scripts as example, 
which includes at least:
- a function with name 'factor_importance' that takes X and y as argument and one optional argument norm
- a `__name__` and `__fullname__` attribute
- adding the new module name to the `__init_file__.py` file in the folder models

Other models for feature selections have been considered, such as PCA or SVD-based methods or
univariate screening methods (t-test, correlation, etc.). However, some of these models consider either 
only linear relationships, or do not take into account the potential multivariate nature of the data structure 
(e.g., higher order interaction between variables). Note that not all included models are completely generalizable, 
such as Bayesian regression and Spearman ranking given their dependence on monotonic functional behavior.

Since most models have some limitations or rely on certain data assumptions, it is important to consider a variety 
of techniques for feature selection and to apply model cross-validations.

## License

MIT License

Copyright (c) 2022 Sebastian Haan
