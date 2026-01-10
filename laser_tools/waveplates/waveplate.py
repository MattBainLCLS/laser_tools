import numpy as np
import scipy.constants as const
from scipy.interpolate import CubicSpline

class Waveplate:

    min_frequency : float
    max_frequency : float
    frequencies : np.ndarray
    retardance : np.ndarray

    meta_data : dict
    

    def __init__(self, data_path: str):

        self.data_path = data_path

        mirror_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        wavelengths = mirror_data[:, 0]
        self.reflectivity_p = mirror_data[:, 1]

        self.frequencies = np.divide(const.c, wavelengths*1E-9)

        if np.any(np.diff(self.frequencies) < 0):
            self.frequencies = np.flip(self.frequencies)
            self.retardance = np.flip(mirror_data[:, 1])

        self.min_frequency = np.min(self.frequencies)
        self.max_frequency = np.max(self.frequencies)

        self.interpolator = CubicSpline(self.frequencies, self.retardance, extrapolate = False)

    # Todo implement waveplate operation using Jones Calculus

    # Todo What would be really cool is if this also implemented the amount of material propagation to calculated GDD
    
