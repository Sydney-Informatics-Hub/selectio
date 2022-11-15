# Custom utility functions for visualisation and modeling

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy


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


def plot_feature_correlation_spearman(X, feature_names, outpath, show = False):
	"""
	Plot feature correlations using Spearman correlation coefficients.
	Feature correlations are automatically clustered using hierarchical clustering.

	Result figure is automatically saved in specified path.

	Input:
		X: data array
		feature names: list of feature names
		outpath: path to save plot
		show: if True, interactive matplotlib plot is shown
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
	corr = spearmanr(X).correlation
	corr_linkage = hierarchy.ward(corr)
	dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90)
	dendro_idx = np.arange(0, len(dendro['ivl']))

	# Plot results:
	pos = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
	ax2.set_xticks(dendro_idx)
	ax2.set_yticks(dendro_idx)
	ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
	ax2.set_yticklabels(dendro['ivl'])
	fig.colorbar(pos, ax = ax2)
	fig.tight_layout()
	plt.savefig(os.path.join(outpath, 'Feature_Correlations_Hierarchical_Spearman.png'), dpi = 300)
	if show:
		plt.show()


def plot_importance_map(dfscores, modelnames, model_fullnames, outpath, show = False):
	"""
	Generates importance matrix plot and saves in output path

	Input:
		dfscores: pandas dataframe with score results
		feature_names: list of fetaure names, must match order in dfscores
		outpath: output directory
		show: boolean, if True shows matplotlib plot
	"""
	columns = ['score_' + modelname for modelname in modelnames]
	columns.append('score_combined')
	mod_fullnames = model_fullnames + ['Combined Score']
	# sort dfscore by column score_combined
	dfscores = dfscores.sort_values(by = 'score_combined', ascending = False)

	# Get sorted value array
	array = dfscores[columns].values # shape (n_features, n_models)
	# Sorted feature names
	feature_names = dfscores['name_features'].values

	# Plot heatmap
	# alternative colormaps: cmap="Wistia"
	fig, ax = plt.subplots(1, figsize=(10, len(feature_names)/len(mod_fullnames) * 6))
	im, _ = heatmap(array, feature_names, mod_fullnames, ax=ax,
					cmap="YlOrRd", cbarlabel="Importance Score")
	_ = annotate_heatmap(im, valfmt="{x:.2f}", size=7)
	plt.tight_layout()
	fname_out = f'Feature_importance_map.png'
	plt.savefig(os.path.join(outpath, fname_out), dpi = 300)
	if show:
		plt.show()
	plt.close('all')


def plot_importance_panels(dfscores, modelnames, model_fullnames, outpath, show = False):
	"""
	Plot all feature importance scores in separate panels

	Input:
		dfscores: pandas dataframe with score results
		feature_names: list of fetaure names, must match order in dfscores
		outpath: output directory
		show: boolean, if True shows matplotlib plot
	"""
	feature_names = dfscores['name_features'].values
	fig, ax = plt.subplots(3,2, figsize = (10,8))
	j =0 
	for i in range(6):
		modelname = modelnames[i]
		#model_fullname = _list_models[i].__fullname__
		model_fullname = model_fullnames[i]
		scores = dfscores['score_' + modelname].values
		sorted_idx = scores.argsort()
		ypos = np.arange(len(scores))
		if i >= 3:
			i -= 3
			j = 1
		bar = ax[i, j].barh(ypos, scores[sorted_idx], tick_label = np.asarray(feature_names)[sorted_idx], align='center')
		gradientbars(bar, scores[sorted_idx])
		ax[i, j].set_title(f'{model_fullname}')	
	ax[2, 0].set_xlabel('Feature Importance Score')	
	ax[2, 1].set_xlabel('Feature Importance Score')
	plt.tight_layout()
	fname_out = f'Feature_importances_all.png'
	plt.tight_layout()
	plt.savefig(os.path.join(outpath, fname_out), dpi = 300)
	if show:
		plt.show()
	plt.close('all')


## heatmap functions

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
