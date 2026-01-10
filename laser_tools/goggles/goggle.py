from scipy.interpolate import CubicSpline
import scipy.constants as const
import numpy as np
import copy
import os

class Goggle(): # Parent class for a set of laser goggles

    min_frequency: float
    max_frequency: float
    frequencies: np.ndarray
    optical_densities: np.ndarray

    interpolator: CubicSpline

    def __init__(self, data_path: str):

        self.load_data(data_path)

    def load_data(self, data_file: str, unit = "nm"):

        DATA_PATH = os.path.join(os.path.dirname(__file__), "data", data_file)

        goggle_data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)

        match unit:
            case "Hz":
                self.frequencies = goggle_data[:,0]
            case "nm":
                self.frequencies = np.divide(const.c, goggle_data[:,0]*1E-9)
            case "m":
                self.frequencies = np.divide(const.c, goggle_data[:,0])

        self.optical_densities = goggle_data[:,1]
        # As is optical density (e.g unitless due to ratio) no Jacobian required

        # Checks for whether monotonically increasing or not and corrects. Required for subsequent interpolation
        if np.any(np.diff(self.frequencies) < 0):
            self.frequencies = np.flip(self.frequencies)
            self.optical_densities = np.flip(self.optical_densities)

        self.min_frequency = np.min(self.frequencies)
        self.max_frequency = np.max(self.frequencies)

        self.interpolator = CubicSpline(self.frequencies, self.optical_densities, extrapolate = False)

    def evaluate_OD(self, frequency):
        return self.interpolator(frequency)

    def attenuate(self, frequency, intensities):
        attenuated_intensities = np.copy(intensities)
        valid_indices = self.valid_indices(frequency)
        ODs = self.evaluate_OD(frequency[valid_indices])
        attenuated_intensities[valid_indices] = np.divide(attenuated_intensities[valid_indices], np.power(10, ODs))
        return attenuated_intensities

    def attenuate_pulse(self, pulse): # Should this be an independent method that takes a goggle and a pulse as arguments?

        attenuated_pulse = copy.deepcopy(pulse) # Avoid modifying the original

        valid_indices = self.valid_indices(attenuated_pulse.frequency_axis)

        ODs = self.evaluate_OD(attenuated_pulse.frequency_axis[valid_indices])
        Ef = attenuated_pulse.Ef
        Ef[valid_indices] = np.divide(Ef[valid_indices], np.power(10, ODs))
        attenuated_pulse.Ef = Ef

        return attenuated_pulse

    def valid_indices(self, frequencies):
        return np.argwhere((frequencies > self.min_frequency) & (frequencies < self.max_frequency))
