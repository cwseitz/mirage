import numpy as np; import pandas as pd; import pims
import matplotlib.pyplot as plt
from ..video import anno_blob, anno_scatter
from ..video import plot_end
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
from matplotlib.ticker import FormatStrFormatter

def detect_blobs(pims_frame,
				min_sig=1,
				max_sig=3,
				num_sig=5,
				blob_thres=0.1,
				peak_thres_rel=0.1,
				r_to_sigraw=3,
				pixel_size = .1084,
				diagnostic=True,
				pltshow=True,
				plot_r=True,
				blob_marker='^',
				blob_markersize=10,
				blob_markercolor=(0,0,1,0.8),
				truth_df=None):
	"""

	Detect blobs for each frame.

	Parameters
	----------
	pims_frame : pims.Frame object
		Each frame in the format of pims.Frame.
	min_sig : float, optional
		As 'min_sigma' argument for blob_log().
	max_sig : float, optional
		As 'max_sigma' argument for blob_log().
	num_sig : int, optional
		As 'num_sigma' argument for blob_log().
	blob_thres : float, optional
		As 'threshold' argument for blob_log().
	peak_thres_rel : float, optional
		Relative peak threshold [0,1].
		Blobs below this relative value are removed.
	r_to_sigraw : float, optional
		Multiplier to sigraw to decide the fitting patch radius.
	pixel_size : float, optional
		Pixel size in um. Used for the scale bar.
	diagnostic : bool, optional
		If true, run the diagnostic.
	pltshow : bool, optional
		If true, show diagnostic plot.
	plot_r : bool, optional
		If True, plot the blob boundary.
	truth_df : DataFrame or None. optional
		If provided, plot the ground truth position of the blob.

    Examples
    --------
	>>> import pims
	>>> from trackit.track import detect_blobs, detect_blobs_batch
	>>> frames = pims.open('cellquantifier/data/simulated_cell.tif')
	>>> detect_blobs(frames[0])

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plot.


	"""

	# """
	# ~~~~~~~~~~~~~~~~~Detection using skimage.feature.blob_log~~~~~~~~~~~~~~~~~
	# """

	frame = pims_frame
	blobs = blob_log(frame,
					 min_sigma=min_sig,
					 max_sigma=max_sig,
					 num_sigma=num_sig,
					 threshold=blob_thres)

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df and update it~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak']
	blobs_df = pd.DataFrame([], columns=columns)
	blobs_df['x'] = blobs[:, 0]
	blobs_df['y'] = blobs[:, 1]
	blobs_df['sig_raw'] = blobs[:, 2]
	blobs_df['r'] = blobs[:, 2] * r_to_sigraw
	blobs_df['frame'] = pims_frame.frame_no
	# """
	# ~~~~~~~Filter detections at the edge~~~~~~~
	# """
	blobs_df = blobs_df[(blobs_df['x'] - blobs_df['r'] > 0) &
				  (blobs_df['x'] + blobs_df['r'] + 1 < frame.shape[0]) &
				  (blobs_df['y'] - blobs_df['r'] > 0) &
				  (blobs_df['y'] + blobs_df['r'] + 1 < frame.shape[1])]
	for i in blobs_df.index:
		x = int(blobs_df.at[i, 'x'])
		y = int(blobs_df.at[i, 'y'])
		r = int(round(blobs_df.at[i, 'r']))
		blob = frame[x-r:x+r+1, y-r:y+r+1]
		blobs_df.at[i, 'peak'] = blob.max()

	# """
	# ~~~~~~~Filter detections below peak_thres_abs~~~~~~~
	# """

	peak_thres_abs = blobs_df['peak'].max() * peak_thres_rel
	blobs_df = blobs_df[(blobs_df['peak'] > peak_thres_abs)]

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~Print detection summary~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	if len(blobs_df)==0:
		print("\n"*3)
		print("##############################################")
		print("ERROR: No blobs detected in this frame!!!")
		print("##############################################")
		print("\n"*3)
		return pd.DataFrame(np.array([])), np.array([])
	else:
		print("Det in frame %d: %s" % (pims_frame.frame_no, len(blobs_df)))

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Diagnostic~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	plt_array = []
	if diagnostic:
		fig, ax = plt.subplots(figsize=(9,9))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~Annotate the blobs~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		ax.imshow(frame, cmap="gray", aspect='equal')
		anno_blob(ax, blobs_df, marker=blob_marker, markersize=blob_markersize,
				plot_r=plot_r, color=blob_markercolor)

		# """
		# ~~~~~~~~~~~~~~~~~~~Annotate ground truth if needed~~~~~~~~~~~~~~~~~~~
		# """
		if isinstance(truth_df, pd.DataFrame):
			anno_scatter(ax, truth_df, marker='o', color=(0,1,0,0.8))

		# """
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Add scale bar~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# """
		font = {'family': 'arial', 'weight': 'bold','size': 16}
		scalebar = ScaleBar(pixel_size, 'um', location = 'upper right',
			font_properties=font, box_color = 'black', color='white')
		scalebar.length_fraction = .3
		scalebar.height_fraction = .025
		ax.add_artist(scalebar)

		plt_array = plot_end(fig, pltshow)

	return blobs_df, plt_array


def detect_blobs_batch(pims_frames,
			min_sig=1,
			max_sig=3,
			num_sig=5,
			blob_thres=0.1,
			peak_thres_rel=0.1,
			r_to_sigraw=3,
			pixel_size = 108.4,
			diagnostic=False,
			pltshow=False,
			plot_r=True,
			blob_marker='^',
			blob_markersize=10,
			blob_markercolor=(0,0,1,0.8),
			truth_df=None):

	"""
	Detect blobs for the whole movie.

	Parameters
	----------
	See detect_blobs().

	Returns
	-------
	blobs_df : DataFrame
		columns = ['frame', 'x', 'y', 'sig_raw', 'r',
					'peak', 'mass', 'mean', 'std']
	plt_array :  ndarray
		ndarray of diagnostic plots.

	Examples
	--------
	>>> import pims
	>>> from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	>>> frames = pims.open('cellquantifier/data/simulated_cell.tif')
	>>> detect_blobs_batch(frames, diagnostic=0)

	"""

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Prepare blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	columns = ['frame', 'x', 'y', 'sig_raw', 'r', 'peak', 'mass', 'mean', 'std']
	blobs_df = pd.DataFrame([], columns=columns)
	plt_array = []

	# """
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Update blobs_df~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# """

	for i in range(len(pims_frames)):
		current_frame = pims_frames[i]
		fnum = current_frame.frame_no
		if isinstance(truth_df, pd.DataFrame):
			current_truth_df = truth_df[truth_df['frame'] == fnum]
		else:
			current_truth_df = None

		tmp, tmp_plt_array = detect_blobs(pims_frames[i],
					   min_sig=min_sig,
					   max_sig=max_sig,
					   num_sig=num_sig,
					   blob_thres=blob_thres,
					   peak_thres_rel=peak_thres_rel,
					   r_to_sigraw=r_to_sigraw,
					   pixel_size=pixel_size,
					   diagnostic=diagnostic,
					   pltshow=pltshow,
					   plot_r=plot_r,
					   blob_marker=blob_marker,
					   blob_markersize=blob_markersize,
					   blob_markercolor=blob_markercolor,
					   truth_df=current_truth_df)
		blobs_df = pd.concat([blobs_df, tmp], sort=True)
		plt_array.append(tmp_plt_array)

	blobs_df.index = range(len(blobs_df))
	plt_array = np.array(plt_array)

	return blobs_df, plt_array
