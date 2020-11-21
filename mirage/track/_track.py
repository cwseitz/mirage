import numpy as np
import pandas as pd
import trackpy as tp
import sys
from bamboo.metrics import fit_msd

def get_d_values(traj_df, im, divide_num):

	"""
	Returns a modififed traj_df with an extra column for each particles
	diffusion coefficient

	Parameters
	----------
	traj_df : DataFrame
		DataFrame with column for frame number and x,y particle coordinates

	im: DataFrame
		result of trackpy.imsd

	divide_num: int
		Fraction of the temporal information to keep e.g 5 for first 1/5
		of the total time

	Returns
	-------
	traj_df : DataFrame object
		DataFrame with added column for diffusion coefficient and alpha value

	"""

	n = int(round(len(im.index)/divide_num))
	im = im.head(n)

	#get diffusion coefficient of each particle
	particles = im.columns
	for particle in particles:

		# Remove NaN, Remove non-positive value before calculate log()
		msd = im[particle].dropna()
		msd = msd[msd > 0]

		if len(msd) > 2: # Only fit when msd has more than 2 data points
			x = msd.index.values
			y = msd.to_numpy()
			y = y*1e6 #convert to nm
			popt = fit_msd(x, y)

			traj_df.loc[traj_df['particle'] == particle, 'D'] = popt[0]
			traj_df.loc[traj_df['particle'] == particle, 'alpha'] = popt[1]

	return traj_df


def track_blobs(blobs_df,
			    search_range=3,
				memory=5,
				pixel_size=.1084,
				frame_rate=3.3,
				divide_num=5,
				filters=None,
				do_filter=False):

	"""
	Wrapper for trackpy library functions (assign detection instances to particle trajectories)

	Parameters
	----------
	blobs_df : DataFrame
		DataFrame with column for frame number and x,y particle coordinates

	search_range: int
		the maximum distance a particle can move between frames and still be tracked

	memory: int
		the number of frames to remember a particle that has disappeared

	filters: dict
		a dictionary of filters to apply to the blob DataFrame

	pixel_size: float
		the pixel_size of the images in microns/pixel

	frame_rate: float
		the frequency of the time-series acquisition in frames/sec

	divide_num: int
		The number used to divide the msd curves

	Returns
	-------
	blobs_df_tracked : DataFrame object
		DataFrame with added column for particle number

	Examples
	--------
	>>> from cellquantifier import data
	>>> from cellquantifier.smt.detect import detect_blobs, detect_blobs_batch
	>>> from cellquantifier.smt.fit_psf import fit_psf, fit_psf_batch
	>>> from cellquantifier.smt.track import track_blobs

	>>> frames = data.simulated_cell()
	>>> blobs_df, det_plt_array = detect_blobs_batch(frames)
	>>> psf_df, fit_plt_array = fit_psf_batch(frames, blobs_df)
	>>> blobs_df, im = track_blobs(psf_df, min_traj_length=10)

	"""

	blobs_df = blobs_df.dropna(subset=['x', 'y', 'frame'])

	# """
	# ~~~~~~~~~~~Apply filters, Link Trajectories~~~~~~~~~~~~~~
	# """

	if do_filter:

		print("######################################")
		print("Filtering out suspicious data points")
		print("######################################")

		blobs_df = blobs_df[blobs_df['dist_err'] < filters['MAX_DIST_ERROR']]
		blobs_df = blobs_df[blobs_df['delta_area'] < filters['MAX_DELTA_AREA']]
		blobs_df = blobs_df[blobs_df['sigx_to_sigraw'] < filters['SIG_TO_SIGRAW']]
		blobs_df = blobs_df[blobs_df['sigy_to_sigraw'] < filters['SIG_TO_SIGRAW']]
		blobs_df = tp.link_df(blobs_df, search_range=search_range, memory=memory)
		blobs_df = tp.filter_stubs(blobs_df, 5)
		blobs_df = blobs_df.reset_index(drop=True)

	else:
		blobs_df = tp.link_df(blobs_df, search_range=search_range, memory=memory)
		blobs_df = blobs_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Check if DataFrame is empty~~~~~~~~~~~~~
	# """

	if blobs_df.empty:
		print('\n***Trajectories num is zero***\n')
		return blobs_df, blobs_df

	# """
	# ~~~~~~~~~~~Get dA/A~~~~~~~~~~~~~
	# """

	print("######################################")
	print("Calculating delta_area")
	print("######################################")

	blobs_df = blobs_df.sort_values(['particle', 'frame'])
	blobs_df['delta_area'] = np.abs((blobs_df.groupby('particle')['area'].apply(pd.Series.pct_change)))
	blobs_df['delta_area'] = blobs_df['delta_area'].fillna(0)

	# """
	# ~~~~~~~~~~~Get Individual Particle D Values~~~~~~~~~~~~~~
	# """

	blobs_df_cut = blobs_df[['frame', 'x', 'y', 'particle']]
	blobs_df_cut = blobs_df_cut.apply(pd.to_numeric)
	im = tp.imsd(blobs_df_cut, mpp=pixel_size, fps=frame_rate, max_lagtime=np.inf)

	blobs_df = get_d_values(blobs_df, im, divide_num)
	blobs_df = blobs_df.apply(pd.to_numeric)

	return blobs_df, im
