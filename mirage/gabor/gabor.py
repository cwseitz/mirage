import numpy as np
import matplotlib.pyplot as plt

def harmonic(X, w0):

    """
    Returns a 2D harmonic function parameterized by a wave vector
    """

    x = X[0]; y = X[1]
    w_x, w_y = w0
    psi_real = np.cos(w_x*x + w_y*y)
    psi_imag = np.sin(w_x*x + w_y*y)

    return psi_real, psi_imag


def gaussian_2d(X, r0, sigma, phi):

    """
    2D Gaussian function

    Parameters
    ----------

    References
    ----------
    Tai Sing Lee, Image Representation with 2D Gabor Wavelets, IEEE 1996

    Returns
    -------

    """

    x, y = X; x0, y0 = r0
    sigma_x, sigma_y = sigma
    a = (np.cos(phi)*(x-x0) + np.sin(phi)*(y-y0))/sigma_x
    b = (-np.sin(phi)*(x-x0) + np.cos(phi)*(y-y0))/sigma_y
    gaussian_2d = np.exp(-(a**2 + b**2)/2)

    return gaussian_2d

def gabor(X, r0, w0, sigma, phi):

    psi_a = gaussian_2d(X, r0, sigma, phi)
    psi_b_real, psi_b_imag = harmonic(X, w0)

    return psi_a*psi_b_real

def gabor_frequency(X, w0, sigma, phi):
    return gaussian_2d(X, w0, 1/np.array(sigma), phi)

p = np.linspace(-50, 50, 1000)
q = np.linspace(-50, 50, 1000)
X = np.meshgrid(p, p)
Y = np.meshgrid(q, q)

sigma = [5,3]
fig, ax = plt.subplots(1,2)

#Generate points on a circle
c = np.linspace(0, 2*np.pi, 5)

for i in c:

    r0 = [2*np.cos(i), 2*np.sin(i)]
    w = [np.cos(i),np.sin(i)] #frequency domain coordinates
    g1 = gabor(X, r0, w, sigma, i)

    #g2 = gabor_frequency(Y, w, sigma, i)
    ax[0].imshow(g1, cmap='gray')
    #ax[1].imshow(g2, cmap='gray')

plt.show()
