import numpy as np
from .. import psf

def add_psfs(frame,
			 traj_df,
			 ex_wavelen=488,
			 em_wavelen=520,
			 num_aperture=1.22,
			 refr_index=1.333,
			 pinhole_rad=.55,
			 pinhole_shape='round'):

	"""Adds point-spread-functions (PSF) to a single frame.

	Parameters
	----------

	frame : ndarray
		a single frame to be populated
	traj_df : DataFrame
		DataFrame containing particle trajectories e.g. x, y, (z) columns
	ex_wavelen : float
		excitation wavelength, used to compute the PSF
	em_wavelen : float
		emission wavelength, used to compute the PSF
	num_aperture : float
		numerical aperature of the objective, used to compute the PSF
	refr_index : float
		refractive index of the imaging medium, used to compute the PSF
	pinhole_rad : float
		radius of the pinhole, used to compute the PSF
	pinhole_shape : str
		shape of the pinhole, used to compute the PSF

	"""

	args = dict(ex_wavelen=ex_wavelen, em_wavelen=em_wavelen,
				num_aperture=num_aperture, refr_index=refr_index,
				pinhole_radius=pinhole_rad, pinhole_shape=pinhole_shape)

	obsvol = psf.PSF(psf.GAUSSIAN | psf.CONFOCAL, **args)
	sigma_px, sigma_um = obsvol.sigma.ou
	particles = traj_df['particle'].unique()

	xsize, ysize = frame.shape[2:4]
	x = np.linspace(0, xsize-1, xsize, dtype=np.int)
	y = np.linspace(0, ysize-1, ysize, dtype=np.int)
	x, y = np.meshgrid(x, y)

	for particle in particles:

		pos = traj_df.loc[traj_df['particle'] == particle, ['x','y']].to_numpy()
		pos += np.round(frame.shape[3]/2); x0,y0 = pos[0]
		_psf = np.exp(-((x-x0)**2/(2*sigma_um**2) + (y-y0)**2/(2*sigma_um**2)))

		nphotons = traj_df.loc[traj_df['particle'] == particle, \
									   'photons'].to_numpy()
		_psf = _psf*nphotons[0]; frame += _psf

	return frame

def add_psfs_batch(frames,
				   traj_df,
				   ex_wavelen=488,
				   em_wavelen=520,
				   num_aperture=1.22,
				   refr_index=1.333,
				   pinhole_rad=.55,
				   pinhole_shape='round'):

	"""Adds point-spread-functions (PSF) to a multiple frames.

	Parameters
	----------

	frame : ndarray
		a single frame to be populated
	traj_df : DataFrame
		DataFrame containing trajectories e.g. x, y, (z) columns, optional
	ex_wavelen : float
		excitation wavelength, used to compute the PSF, optional
	em_wavelen : float
		emission wavelength, used to compute the PSF, optional
	num_aperture : float
		numerical aperature of the objective, used to compute the PSF, optional
	refr_index : float
		refractive index of the medium, used to compute the PSF, optional
	pinhole_rad : float
		radius of the pinhole, used to compute the PSF, optional
	pinhole_shape : str
		shape of the pinhole, used to compute the PSF, optional

	"""

	nframes = len(frames)
	for n in range(nframes):
		this_df = traj_df.loc[traj_df['frame'] == n]
		frames[n] = add_psfs(frames[n],
							 this_df,
							 ex_wavelen=ex_wavelen,
							 em_wavelen=em_wavelen,
							 num_aperture=num_aperture,
							 refr_index=refr_index,
							 pinhole_rad=pinhole_rad,
							 pinhole_shape=pinhole_shape)

	return frames

def add_noise(frame, sigma_dark=10, bit_depth=8, baseline=0):

	"""Add dark noise to a single frame, quantize intensity

	Parameters
	----------

	frame : ndarray,
		a single frame
	sigma_dark : float, optional
		standard deviation of the gaussian noise distribution
	bit_depth : float, optional
		bit depth of the frame
	baseline : float, optional
		intensity baseline of the image

	"""

	# Add dark noise
	frame = np.random.normal(scale=sigma_dark, \
							  size=frame.shape) + frame
	frame += baseline

	# Set negative values to zero
	frame = frame.clip(min=0)

	return frame


def add_noise_batch(frames,
					sigma_dark=10,
					bit_depth=8,
					baseline=0):

	"""Add dark noise to a multiple frames

	Parameters
	----------

	frames : ndarray,
		a stack of frames
	sigma_dark : float, optional
		standard deviation of the gaussian noise distribution
	bit_depth : float, optional
		bit depth of the frame
	baseline : float, optional
		intensity baseline of the image

	"""

	nframes = len(frames)
	for n in range(nframes):
		frames[n] = add_noise(frames[n],
							  sigma_dark=sigma_dark,
							  bit_depth=bit_depth,
							  baseline=baseline)

	return frames

def add_photon_stats(df, exp_time=1, photon_rate=1, add_noise=True):

	"""Add photon numbers to DataFrame

	Parameters
	----------

	df : DataFrame,
		DataFrame containing frame, x, y, (z) columns
	exp_time : float, optional
		exposure time in arbitrary units
	photon_rate : float, optional
		rate parameter for Poisson distribution (photon statistics)
	add_noise : bool, optional
		whether or not to add shot noise to the image

	"""

	if not add_noise:
		df = df.assign(photons=1)

	else:
		nrecords = df.shape[0]
		lam=exp_time*photon_rate*np.ones(nrecords)
		nphotons = np.random.poisson(lam=exp_time*photon_rate,size=nrecords)
		df['photons'] = nphotons

	return df
