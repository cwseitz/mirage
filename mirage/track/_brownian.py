import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.stats import norm
from numpy.random import normal, uniform

def get_trajectory_1(nframes, dcoeff, origin=None):

	"""
	Build a trajectory where each displacement is selected from a gaussian
	distribution N(m, sigma) where sigma is constant. Displacements are measured
	relative to preceding position

	Use this method when msd = <r(t+n) - r(t)> (individual msd)

	Parameters
	----------
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	traj_df = pd.DataFrame(columns=columns)
	traj_df['frame'] = np.arange(0, nframes, 1)
	traj_df['sigma'] = np.sqrt(2*dcoeff)
	traj_df['dx'] = normal(size=nframes, scale=traj_df['sigma'])
	traj_df['dy'] = normal(size=nframes, scale=traj_df['sigma'])

	if origin is None:
		x0,y0 = (0,0)
	else:
		x0, y0 = origin

	traj_df['x'] = traj_df['dx'].cumsum() + x0
	traj_df['y'] = traj_df['dy'].cumsum() + y0

	return traj_df

def get_trajectory_2(nframes, dcoeff, origin=None):

	"""
	Build a trajectory where each displacement is selected from a gaussian
	distribution N(m, sigma) where sigma is a function of time. Displacements
	are measured relative to the origin

	Use this method when msd = <r(t) - r(0)> (ensemble msd)

	Parameters
	----------
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	traj_df = pd.DataFrame(columns=columns)
	traj_df['frame'] = np.arange(0, nframes, 1)
	traj_df['sigma'] = np.sqrt(2*dcoeff*traj_df['frame'])
	traj_df['dx'] = normal(size=nframes, scale=traj_df['sigma'])
	traj_df['dy'] = normal(size=nframes, scale=traj_df['sigma'])

	if origin is None:
		x0,y0 = (0,0)
	else:
		x0, y0 = origin

	traj_df['x'] = traj_df['dx'] + x0
	traj_df['y'] = traj_df['dy'] + y0

	return traj_df

def get_trajectory_3(traj_df=None,
					 nparticles=3,
					 nsamples=5,
					 deg_freedom=3,
					 env_temp=293,
					 env_visc=1,
					 particle_rad=1,
					 particle_mass=1,
					 dt=.1,
					 damped=True,
					 corr_mat_3d=None,
					 std=None):

	"""
	Generate an instance of Brownian motion (i.e. the Wiener process) by
	solving the Langevin equation for damped Brownian motion via Euler's method

	Arguments
	---------
	traj_df : DataFrame
			  Contains frame, x,y,particle columns
			  which must be prepopulated with particles and their origins,

	v0 : ndarray
		 The initial velocities of the Brownian motion. If not specified
		 particles begin at rest

	nsamples : int
		The number of time steps

	dt : float
		The time step period i.e. 1/samplerate

	Example
	-------

	"""

	def solve(a, damp_ratio, dt=0.1):
		n = len(a[0])
		v = np.zeros_like(a)
		for i in range(len(a)-1):
			for j in range(n):
				v[i+1, j] = v[i,j] + a[i, j]*dt - damp_ratio*v[i, j]*dt
		return v

	#
	#~~~~~~~Calculate friction coefficient~~~~~~~~~~
	#

	if not damped:
		fric_coeff=0
	else:
		fric_coeff = 6*np.pi*env_visc*particle_rad #stokes law

	damp_ratio = fric_coeff/particle_mass

	#
	#~~~~~~~Variance of stochastic force distribution~~~~~~~~~~
	#

	k_b = 1.38e-23
	xi_var = 2*fric_coeff*k_b*env_temp
	diff_coeff = (k_b*env_temp/fric_coeff)*1e18

	#
	#~~~~~~~Pretty print the parameters~~~~~~~~~~
	#

	print('T: %.2E K' % env_temp)
	print('viscosity: %.2E Pa-s' % env_visc)
	print('radius: %.2E m' % particle_rad)
	print('mass: %.2E kg' % particle_mass)
	print('force variance: %.2E' % xi_var)
	print('friction coeff: %.2E' % fric_coeff)
	print('diffusion coeff: %.2E' % diff_coeff)

	#
	#~~~~~~~Setup dataframe~~~~~~~~~~
	#

	if traj_df is None:
		columns = ['frame', 'x', 'y', 'z', 'particle']
		traj_df = pd.DataFrame(columns=columns)

	samples = np.arange(0,nsamples,1)
	particles = np.arange(0,nparticles,1)
	traj_df['frame'] = np.repeat(samples, nparticles)
	traj_df['particle'] = np.tile(particles, nsamples)

	traj_df = traj_df.assign(a_x=0,a_y=0,a_z=0,\
							 dv_x=0,dv_y=0,dv_z=0,\
							 dx=0,dy=0,dz=0)

	if 'v' not in traj_df:
		traj_df = traj_df.assign(v_x=0)
		traj_df = traj_df.assign(v_y=0)
		traj_df = traj_df.assign(v_z=0)
	#
	#~~~~~~~Get forces, add to dataframe~~~~~~~~~~
	#

	corr_movie = []
	for n in range(nsamples):
		f = norm.rvs(size=(nparticles, 3), scale=np.sqrt(xi_var))
		if corr_mat_3d is not None and std is not None:
			cov_mat = corr2cov(corr_mat_3d[n], std)
			if not is_pd(cov_mat):
				warnings.warn('Warning: covariance matrix is not pos-definite.\
								converting to nearest pos-definite matrix')
				cov_mat = nearest_pd(cov_mat)

			#
			#~~~~~~~Add to cov_mat movie~~~~~~~~~~
			#

			f = corr_rand(f, cov_mat)

		#update the dataframe with forces
		traj_df.loc[traj_df['frame'] == n, 'a_x'] = f[:,0]/particle_mass
		traj_df.loc[traj_df['frame'] == n, 'a_y'] = f[:,1]/particle_mass
		traj_df.loc[traj_df['frame'] == n, 'a_z'] = f[:,2]/particle_mass

	#
	#~~~~~~~Solve the Langevin Equation~~~~~~~~~~
	#

	for n in range(nparticles):

		this_df = traj_df.loc[traj_df['particle'] == n]
		a = this_df[['a_x','a_y','a_z']].to_numpy(dtype='float64')
		v = solve(a, damp_ratio, dt=dt)
		dv = np.diff(v, axis=0, prepend=np.zeros((1,3)))
		dr = dv*dt
		r = np.cumsum(dr, axis=0)

		traj_df.loc[traj_df['particle'] == n, ['v_x','v_y','v_z']] = v
		traj_df.loc[traj_df['particle'] == n, ['dv_x','dv_y','dv_z']] = dv
		traj_df.loc[traj_df['particle'] == n, ['x','y','z']] = r
		traj_df.loc[traj_df['particle'] == n, ['dx','dy','dz']] = dr


	return traj_df

def get_trajectories(nparticles,
					 nframes,
					 dcoeff=10,
					 method='individual',
					 origins=None):

	"""Build a set of trajectories based on gaussian distributed displacements

	Parameters
	----------
	framerate: float,
			   frames per second
	"""

	columns = ['frame', 'particle', 'x', 'y','dx','dy']
	df = pd.DataFrame(columns=columns)

	if origins is None:
		origins = [None for n in range(nparticles)]

	for n in range(nparticles):
		if method == 'individual':
			this_df = get_trajectory_1(nframes, dcoeff, origin=origins[n])
		elif method == 'ensemble':
			this_df = get_trajectory_2(nframes, dcoeff, origin=origins[n])

		this_df = this_df.assign(particle=n)
		df = pd.concat([df, this_df])

	return df
