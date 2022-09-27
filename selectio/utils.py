# Custom utility functions for visualisation and modeling

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def gradientbars(bars, data):
	"""
	Helper function for making colorfull bars

	Input:
		bars: list of bars
		data: data to be plotted
	"""
	ax = bars[0].axes
	lim = ax.get_xlim()+ax.get_ylim()
	ax.axis(lim)
	for bar in bars:
		bar.set_zorder(1)
		bar.set_facecolor("none")
		x,y = bar.get_xy()
		w, h = bar.get_width(), bar.get_height()
		grad = np.atleast_2d(np.linspace(0,1*w/max(data),256))
		ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0, norm=mpl.colors.NoNorm(vmin=0,vmax=1))
	ax.axis(lim)


def plot_correlationbar(corrcoefs, feature_names, outpath, fname_out, name_method = None, show = False):
	"""
	Helper function for plotting feature correlation.
	Result plot is saved in specified directory.

	Input:
		corrcoefs: list of feature correlations
		feature_names: list of feature names
		outpath: path to save plot
		fname_out: name of output file (should end with .png)
		name_method: name of method used to compute correlations
		show: if True, show plot
	"""
	sorted_idx = corrcoefs.argsort()
	fig, ax = plt.subplots(figsize = (6,5))
	ypos = np.arange(len(corrcoefs))
	bar = ax.barh(ypos, corrcoefs[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
	gradientbars(bar, corrcoefs[sorted_idx])
	if name_method is not None:	
		plt.title(f'{name_method}')	
	plt.xlabel("Feature Importance")
	plt.tight_layout()
	plt.savefig(os.path.join(outpath, fname_out), dpi = 200)
	if show:
		plt.show()
	plt.close('all')


