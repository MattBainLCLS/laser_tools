from scipy.interpolate import CubicSpline
import scipy.constants as const
import numpy as np
import copy
#import matplotlib.pyplot as plt
import os


#import matplotlib.pyplot as plt

def available():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
    return os.listdir(DATA_PATH)



class Goggle(): # Parent class that

    min_frequency: float
    max_frequency: float
    frequencies: np.ndarray
    optical_densities: np.ndarray

    interpolator: CubicSpline

    def __init__(self):
        pass

    def load_data(self, data_file: str, unit = "nm"):

        DATA_PATH = DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

        goggle_data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)

        match unit:
            case "Hz":
                self.frequencies = goggle_data[:,0]
            case "nm":
                self.frequencies = np.divide(const.c, goggle_data[:,0]*1E-9)
            case "m":
                self.frequencies = np.divide(const.c, goggle_data[:,0])

        self.optical_densities = goggle_data[:,1]

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

class DBY(Goggle):
    def __init__(self):
        self.load_data('DBY.csv')

class C1033(Goggle):
    def __init__(self):
        self.load_data('C1033.csv')

class C1023(Goggle):
    def __init__(self):
        self.load_data('C1023.csv')

class T5H03(Goggle):
    def __init__(self):
        self.load_data('T5H03.csv')

class T5H05(Goggle):
    def __init__(self):
        self.load_data('T5H05.csv')