import numpy as np
import matplotlib.pyplot as plt


def harmonic(X, A, k):

    """
    Returns a 2D harmonic function parameterized by frequency and direction
    """

    x = X[0]; y = X[1]
    k_x, k_y = k
    psi_real = np.cos(k_x*x + k_y*y)
    psi_imag = np.sin(k_x*x + k_y*y)

    return psi_real, psi_imag


def gaussian_2d(X, A, x0, y0, sig_x, sig_y, phi):

    """
    2D Gaussian function
    Parameters
    ----------
    X : 3d ndarray
        X = np.indices(img.shape).
		X[0] is the row indices.
        Y[1] is the column indices.
    A : float
        Amplitude.
    x0 : float
        x coordinate of the center.
    y0 : float
        y coordinate of the center.
    sig_x : float
        Sigma in x direction.
    sig_y : float
        Sigma in x direction.
    phi : float
        Angle between long axis and x direction.
    Returns
    -------
    result_array_2d: 2d ndarray
        2D gaussian.
    """

    x = X[0]
    y = X[1]
    a = (np.cos(phi)**2)/(2*sig_x**2) + (np.sin(phi)**2)/(2*sig_y**2)
    b = -(np.sin(2*phi))/(4*sig_x**2) + (np.sin(2*phi))/(4*sig_y**2)
    c = (np.sin(phi)**2)/(2*sig_x**2) + (np.cos(phi)**2)/(2*sig_y**2)
    result_array_2d = A*np.exp(-(a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2))

    return result_array_2d

X = np.indices((100,100))
g = gaussian_2d(X, 1, 50, 50, 3, 2, np.pi)
h_real, h_imag = harmonic(X, 1, (1,0))
psi = g*h_real

fig, ax = plt.subplots()
ax.imshow(psi, cmap='coolwarm')
plt.show()
