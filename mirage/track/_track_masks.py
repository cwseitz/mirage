
import pandas as pd
import trackpy as tp
import numpy as np

def track_masks(mask_arr,
				memory=5,
				z_filter=0.5,
				search_range=20,
				min_traj_length=10,
				min_size=None,
				do_filter=False):

	"""

	Builds a DataFrame from a movie of masks and runs the
	tracking algorithm on that DataFrame

	Parameters
	----------
	mask_arr : ndarray
		ndarray containing the masks

	search_range: int
		the maximum distance the centroid of a mask can move between frames
		and still be tracked

	memory: int
		the number of frames to remember a mask that has disappeared


	Returns
	-------

	"""

	# """
	# ~~~~~~~~~~~Extract mask_df from mask_arr~~~~~~~~~~~~~~
	# """

	mask_df = pd.DataFrame([])
	nframes = mask_arr.shape[0]

	for i in range(nframes):

		props = regionprops_table(mask_arr[i], properties=('centroid', 'area', 'label'))
		this_df = pd.DataFrame(props).assign(frame=i)
		this_df = this_df.rename(columns={'centroid-0':'x', 'centroid-1':'y'})
		mask_df = pd.concat([mask_df, this_df])

	# """
	# ~~~~~~~~~~~Link Trajectories~~~~~~~~~~~~~~
	# """

	mask_df = tp.link_df(mask_df, search_range=search_range, memory=memory)
	mask_df = mask_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Apply filters~~~~~~~~~~~~~~
	# """

	if do_filter:

		print("######################################")
		print("Filtering out suspicious data points")
		print("######################################")

		grp = mask_df.groupby('particle')['area']
		mask_df['z_score'] = grp.apply(lambda x: np.abs((x - x.mean()))/x.std())
		mask_df = mask_df.loc[mask_df['z_score'] < z_filter]

		if min_size:
			mask_df = mask_df.loc[mask_df['area'] > min_size]

		mask_df = tp.link_df(mask_df, search_range=search_range, memory=memory)
		mask_df = tp.filter_stubs(mask_df, min_traj_length)
		mask_df = mask_df.reset_index(drop=True)

	# """
	# ~~~~~~~~~~~Check if DataFrame is empty~~~~~~~~~~~~~
	# """

	if mask_df.empty:
		print('\n***Trajectories num is zero***\n')
		return

	return mask_df
