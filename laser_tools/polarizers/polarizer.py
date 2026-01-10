import numpy as np
import scipy.constants as const
from scipy.interpolate import CubicSpline

class Polarizer:

    min_frequency : float
    max_frequency : float
    frequencies : np.ndarray
    transmitted_p : np.ndarray # Note need to update to a better word as could be a reflective polarizer
    transmitted_s : np.ndarray # Note need to update to a better word as could be a reflective polarizer

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

    # Todo implement polarizer operation using Jones Calculus
