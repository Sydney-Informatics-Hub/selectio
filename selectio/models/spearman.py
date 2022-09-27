"""
Spearman rank analysis for feature importance

Limitations: assumes monotonic function and cannot model interaction terms
"""

import numpy as np
from scipy.stats import spearmanr

__name__ = 'Spearman'
__fullname__ = 'Spearman Rank'

def factor_importance(X_train, y_train, norm = False):
	"""
	Spearman rank analysis

	Input:
		X: input data matrix with shape (npoints,nfeatures)
		y: target varable with shape (npoints)
		norm: normalize results to maximum feature importance is unity

	Return:
		result: feature importances
	"""
	nfeatures = X_train.shape[1]
	corr = np.zeros(nfeatures)
	for i in range(nfeatures):
		sr = spearmanr(X_train[:,i], y_train)
		if sr.pvalue < 0.01:
			corr[i] = sr.correlation
	if norm:
		corr /= corr.max()
	return corr