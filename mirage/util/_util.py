import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.linalg import eigh, cholesky
from numpy import linalg as la

def normalize_by_frame(frames):

	for i, frame in enumerate(frames):
		frames[i] = frames[i]/frames[i].max()
	return frames

def generate_lattice(nparticles=9, ndim=2, show_lattice=False):

	"""
	Generates a 2D/3D lattice of points to be used as particle origins

	Parameters
	----------

	size: int
		if ndim=2, size should be a square
		if ndim=3, size should be a cube

	Returns
	-------

	lattice_pts : ndarray
		coordinates of the lattice

	"""

	#
	#~~~~~~~Check for errors~~~~~~~~~~
	#

	if ndim == 2:
		if not is_perfect_square(nparticles):
			raise ValueError('nparticles is not a perfect square')
			return
		size = int(np.sqrt(nparticles))
	if ndim == 3:
		if not is_perfect_cube(nparticles):
			raise ValueError('nparticles is not a perfect cube')
			return
		size = int(np.cbrt(nparticles))
	#
	#~~~~~~~Build the lattice~~~~~~~~~~
	#

	lattice_range = list(np.linspace(-1,1,size))
	lattice_pts = list(itertools.product(lattice_range, repeat=ndim))
	lattice_pts = np.array(lattice_pts)

	#
	#~~~~~~~Show the lattice~~~~~~~~~~
	#

	if show_lattice:
		fig = plt.figure()

		if ndim == 2:
			ax = fig.gca()
			ax.plot(lattice_pts[:,0], \
					lattice_pts[:,1], \
					'.',
					linewidth=1)

		if ndim == 3:
			ax = fig.gca(projection='3d')
			ax.plot(lattice_pts[:,0], \
					lattice_pts[:,1], \
					lattice_pts[:,2], \
					'.',
					linewidth=1)

		plt.show()

	return lattice_pts

def is_perfect_square(n):

	"""Check if input is a perfect square
	"""

	if n < 0:
		return False
	root = round(np.sqrt(n))
	return n == root**2

def is_perfect_cube(n):

	"""Check if input is a perfect cube
	"""

	if n < 0:
		return False
	root = round(np.cbrt(n))
	return n == root**3

def nearest_pd(A, show=False):

	"""Find the nearest positive-definite matrix to input
	"""

	B = (A + A.T) / 2
	_, s, V = la.svd(B)

	H = np.dot(V.T, np.dot(np.diag(s), V))

	A2 = (B + H) / 2

	A3 = (A2 + A2.T) / 2

	if is_pd(A3):
		return A3

	spacing = np.spacing(la.norm(A))

	I = np.eye(A.shape[0])
	k = 1
	while not is_pd(A3):
		mineig = np.min(np.real(la.eigvals(A3)))
		A3 += I * (-mineig * k**2 + spacing)
		k += 1

	if show:
		fig, ax = plt.subplots(1,2)
		ax[0].imshow(A)
		ax[1].imshow(A3)
		plt.show()

	return A3

def is_pd(B):

	"""Returns true when input matrix is positive-definite, via Cholesky"""

	try:
		_ = la.cholesky(B)
		return True
	except la.LinAlgError:
		return False

def corr2cov(corr, std):
	"""
	convert correlation matrix to covariance matrix given standard deviation

	Parameters
	----------
	corr : array_like, 2d
		correlation matrix, see Notes
	std : array_like, 1d
		standard deviation

	Returns
	-------
	cov : ndarray (subclass)
		covariance matrix

	Notes
	-----
	This function does not convert subclasses of ndarrays. This requires
	that multiplication is defined elementwise. np.ma.array are allowed, but
	not matrices.
	"""
	corr = np.asanyarray(corr)
	std_ = np.asanyarray(std)
	cov = corr * np.outer(std_, std_)
	return cov

def get_corr_template(x, filt, image):

	"""Get a correlation matrix template

	Parameters
	----------
	x : 2D ndarray
		Grid of particle labels
	filt:
		2D ndarray
		The kernel to use for the template (see example)
	image: 2D ndarray
		The image to use for the template

	Example
	----------
	import numpy as np
	import matplotlib.pyplot as plt
	from util import get_corr_template

	x = [np.arange(1,8,1) + 7*i for i in range(7)]
	image1 = np.zeros((49,49))
	kernel_nn = np.array([[0,1,0],
						  [1,0,1],
						  [0,1,0]])

	template1 = get_corr_template(x, kernel_nn, image1)
	plt.imshow(template1)
	plt.show()

	"""

	def f(values, out):
		nonz = np.nonzero(values)
		a = values[nonz] - 1
		result.append(a.astype(np.int))
		return 0

	result = []
	ndimage.generic_filter(x, f, footprint=filt, \
						   mode='constant', extra_arguments=(result,))

	for i, res in enumerate(result):
		for j in res:
			image[i,j] = 1

	return image

def corr_rand(sample,cov_mat,method='cholesky'):

	"""
	Correlate normally distributed samples of data. Samples should be
	uncorrelated and will be correlated by using the eigenvectors/eigenvalues
	of the covariance matrix

	Arguments
	---------
	cov_mat : 2D array
			The covariance matrix

	sample: 2D array
			Can be of any second dimension but must match cov_mat
			along first dimension

	Example
	-------

	"""

	if method == 'cholesky':
		c = cholesky(cov_mat, lower=True)

	else:
		evals, evecs = eigh(cov_mat)
		c = np.dot(evecs, np.diag(np.sqrt(evals)))

	sample = np.dot(c, sample)

	return sample

def get_corr_2D(nparticles, nframes):

	"""
	Generate the standard NN,3N,4N correlation movie in 2D
	with an exponentiaL decaying correlation matrix.

	Arguments
	---------

	Example
	-------

	"""

	size = int(np.sqrt(nparticles))
	x = [np.arange(1,size+1,1) + size*i for i in range(size)]

	corr_nn = np.zeros((nparticles,nparticles))
	corr_3n = np.zeros((nparticles,nparticles))
	corr_4n = np.zeros((nparticles,nparticles))

	kernel_nn = np.array([[0,1,0],
						  [1,0,1],
						  [0,1,0]])

	kernel_3n = np.array([[1,0,1],
						  [0,0,0],
						  [1,0,1]])

	kernel_4n = np.array([[0,1,1,1,0],
						  [1,0,0,0,1],
						  [1,0,0,0,1],
						  [1,0,0,0,1],
						  [0,1,1,1,0]])

	corr_nn = get_corr_template(x, kernel_nn, corr_nn)
	corr_3n = get_corr_template(x, kernel_3n, corr_3n)
	corr_4n = get_corr_template(x, kernel_4n, corr_4n)

	fig,ax=plt.subplots(1,4, figsize=(30,5))
	ax[0].imshow(corr_nn, cmap='coolwarm')
	ax[1].imshow(corr_3n, cmap='coolwarm')
	ax[2].imshow(corr_4n, cmap='coolwarm')
	ax[3].imshow(corr_nn+corr_3n+corr_4n, cmap='coolwarm')
	plt.savefig('/home/clayton/Desktop/temp/test.png', dpi=500)

	a = corr_nn + corr_3n + corr_4n
	b = np.ones_like(a) - a
	tau1 = 50
	tau2 = 25
	crit_frame = 50
	corr_mat_3d = []
	for i in range(nframes):
		if i < crit_frame:
			tmp = .9*np.ones_like(a)
		else:
			a_t = a*np.exp(-(i-crit_frame)/tau1)
			b_t = b*np.exp(-(i-crit_frame)/tau2)
			tmp = a_t + b_t
			# tmp = .9*np.ones_like(a)
			# tmp = np.eye(*a.shape)
		np.fill_diagonal(tmp, 1)
		corr_mat_3d.append(tmp)

	corr_mat_3d = np.array(corr_mat_3d)
	return corr_mat_3d
