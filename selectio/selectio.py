"""
Automatic feature importance calcuations and feature selections based on multiple methods.

Current models for feature importance supported:
- Spearman rank analysis ('spearman')
- linear and log-scaled Bayesian Linear Regression ('blr')
- Random Forest Permutation Test ('rf')
- Random Decision Trees ('rdt')
- Mutual Information Regression ('mi')
- General correlation coefficients ('xicor')

Selectio return a dataframe and plots with feature importance scores for all models.
The final feature selection is based on a majority voting of all models.

The feature selection score can be either computed directly via the class Fsel, e.g.
from selectio import Fsel
fsel = selectio.Fsel(X, y)
dfres = fsel.score_models()

or integrated into the data processing and plotting via a setings yaml file:
import selectio
selectio.main(fname_settings)


User settings, such as input/output paths and all other options, are set in the settings file 
(Default filename: settings_featureimportance.yaml) 
Alternatively, the settings file can be specified as a command line argument with: 
'-s', or '--settings' followed by PATH-TO-FILE/FILENAME.yaml 
(e.g. python featureimportance.py -s settings_featureimportance.yaml).

The selectio package provides to option to generate simulated data (see simdata.py) and
as well as testing functions (see tests.py)

Other models for feature selections exists such as PCA or SVD-based methods or
univariate screening methods (t-test, correlation, etc.). However, some of these models consider either 
only linear relationships, or do not take into account the potential multivariate nature of the data structure 
(e.g., higher order interaction between variables).

More models can be added in the folder 'models' following the example module structure, which includes at least:
- a function with name 'factor_importance' that takes X and y as argument and one optional argument norm
- the new module name should be added to __init_file__.py 
"""

import os
import sys
import yaml
import shutil
import argparse
import datetime
from types import SimpleNamespace  
import numpy as np
import pandas as pd
import importlib

# import some custom plotting utility functions
from utils import plot_correlationbar, plot_feature_correlation_spearman

# import all models for feature importance calculation
#from models import xicor, blr, rf, rdt, mi, spearman
from models import __all__ as _modelnames
_list_models = []
for modelname in _modelnames:
	module = importlib.import_module('models.'+modelname)
	_list_models.append(module)

"""
for model in _list_models:
	importlib.reload(model)
"""

# Settings default yaml file
_fname_settings = 'settings_featureimportance.yaml'


class Fsel:
	"""
	Auto Feature Selection
	"""
	def __init__(self, X, y):
		
		self.X = X
		self.y = y

		self.nmodels = len(_modelnames)
		self.nfeatures = X.shape[1]

		# Initialise pandas dataframe to save results
		self.dfmodels = pd.DataFrame(columns=['score_' + modelname for modelname in _modelnames])


	def score_models(self):
		"""
		Calculate feature importance for all models and select features

		Return:
			corr_array: shape (nmodels, nfeatures)
		"""
		# Loop over all models and calculate normalized feature scores
		count_select = np.zeros(self.nfeatures).astype(int)
		for i in range(self.nmodels):
			model = _list_models[i]
			modelname = _modelnames[i]
			print(f'Computing scores for model {modelname}...')
			corr = model.factor_importance(self.X, self.y, norm = True)
			self.dfmodels['score_' + modelname] = np.round(corr, 4)
			# Calculate which feature scores accepted
			woe = self.eval_score(corr)
			self.dfmodels['woe_' + modelname] = woe
			count_select += woe
			print(f'Done, {woe.sum()} features selected.')
		
		# Select features based on majority vote of models:
		select = np.zeros(self.nfeatures).astype(int)
		select[count_select >= self.nfeatures/2] = 1
		self.dfmodels['selected'] = select
		return self.dfmodels


	def eval_score(self, score, woe_min = 0.01):
		"""
		Evaluate multi-model feature importance scores and select features based on majority vote
		""" 
		sum_score = score.sum()
		min_score = sum_score * woe_min
		woe = np.zeros_like(score)
		woe[score >= min_score] = 1
		return woe.astype(int)



def main(fname_settings):
	"""
	Main function for running the script.

	Input:
		fname_settings: path and filename to settings file
	"""
	# Load settings from yaml file
	with open(fname_settings, 'r') as f:
		settings = yaml.load(f, Loader=yaml.FullLoader)
	# Parse settings dictinary as namespace (settings are available as 
	# settings.variable_name rather than settings['variable_name'])
	settings = SimpleNamespace(**settings)

	# Verify output directory and make it if it does not exist
	os.makedirs(settings.outpath, exist_ok = True)

	# Read data
	data_fieldnames = settings.name_features + [settings.name_target]
	df = pd.read_csv(os.path.join(settings.inpath, settings.infname), usecols=data_fieldnames)

	# Verify that data is cleaned:
	assert df.select_dtypes(include=['number']).columns.tolist().sort() == data_fieldnames.sort(), 'Data contains non-numeric entries.'
	assert df.isnull().sum().sum() == 0, "Data is not cleaned, please run preprocess_data.py before"


	# Generate Spearman correlation matrix for X
	print("Calculate Spearman correlation matrix...")
	plot_feature_correlation_spearman(df[data_fieldnames].values, data_fieldnames, settings.outpath, show = False)

	X = df[settings.name_features].values
	y = df[settings.name_target].values

	# Generate feature importance scores
	fsel = Fsel(X,y)
	dfres = fsel.score_models()

	dfres['name_features'] = settings.name_features
	print('Features selected: ', dfres.loc[dfres.selected == 1, 'name_features'])
	
	# Save results as csv
	dfres.to_csv(os.path.join(settings.outpath, 'feature-importance_scores.csv'), index_label = 'Feature_index')

	# Plot scores
	print("Generating score plots..")
	for i in range(len(_modelnames)):
		modelname = _modelnames[i]
		try:
			model_label = _list_models[i].__name__
			model_fullname = _list_models[i].__fullname__
		except:
			model_label = modelname
			model_fullname = modelname
		scores = dfres['score_' + modelname].values
		plot_correlationbar(scores, settings.name_features, settings.outpath, f'{model_label}-feature-importance.png', name_method = model_fullname, show = False)


	
if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser(description='Calculating feature importance.')
	parser.add_argument('-s', '--settings', type=str, required=False,
						help='Path and filename of settings file.',
						default = os.path.join('./settings',_fname_settings))
	args = parser.parse_args()

	# Log computational time
	datetime_now = datetime.datetime.now()
	# Run main function
	main(args.settings)
	# print out compute time of main function in seconds
	print('Computational time of main function: {:.2f} seconds'.format((datetime.datetime.now() - datetime_now).total_seconds()))