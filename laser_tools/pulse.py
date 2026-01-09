from unittest import case

import numpy as np
from scipy.signal import hilbert
import scipy.constants as const
import scipy.special
from scipy.interpolate import CubicSpline
from numbers import Number

import laser_tools.goggles

from copy import deepcopy


class RealPulse:

    def __init__(self, N : int, dt : float):

        self.time_axis : np.array = np.empty(shape=(1,), dtype=np.float64)
        self.frequency_axis : np.array = np.empty(shape=(1,), dtype=np.float64)
        self.Et_v: np.array = np.empty(shape=(1,), dtype=np.float64)
        self.Et_h: np.array = np.empty(shape=(1,), dtype=np.float64)
        self.Ef_v: np.array = np.empty(shape=(1,), dtype=np.complex128)
        self.Ef_h: np.array = np.empty(shape=(1,), dtype=np.complex128)

        self.carrier_frequency : float = None

        self.make_axes(N, dt)

        #self._accumulated_phase : np.array = np.empty(shape=(1,), dtype=np.float64)

    def make_axes(self, N : int, dt : float):
        lim = dt*(N/2)
        self.time_axis = np.arange(-lim, lim, dt)
        self.frequency_axis = np.fft.rfftfreq(N, dt)
        self.Et_v: np.array = np.zeros(shape=np.shape(self.time_axis), dtype=np.float64)
        self.Et_h: np.array = np.zeros(shape=np.shape(self.time_axis), dtype=np.float64)
        self.Ef_v: np.array = np.zeros(shape=np.shape(self.frequency_axis), dtype=np.complex128)
        self.Ef_h: np.array = np.zeros(shape=np.shape(self.frequency_axis), dtype=np.complex128)

        self._accumulated_phase_v = np.zeros(np.shape(self.frequency_axis))
        self._accumulated_phase_h = np.zeros(np.shape(self.frequency_axis))

        self.normalization_factor = 2*dt # Normalization factor for rfft


    @property
    def It(self) -> np.array:
        return abs2(hilbert(self.Et_v)) + abs2(hilbert(self.Et_h))

    #@It.setter
    #def It(self, value : np.array):
    #    self.Et = np.sqrt(value)

    @property
    def If(self) -> np.array:
        return abs2(self.Ef_v) + abs2(self.Ef_h)

    #@If.setter
    #def If(self, value : np.array):
    #    self.Ef = np.sqrt(value)

    @property
    def energy(self) -> float: # Returns integrated spectral energy in joules
        return np.trapezoid(self.If, x = self.frequency_axis)

    @energy.setter
    def energy(self, value : float):
        factor = value / self.energy
        self.Ef_v /= factor
        self.Ef_h /= factor

    def forward(self):
        self.Ef_v = np.fft.rfft(self.Et_v) * self.normalization_factor # Add the normalization
        self.Ef_h = np.fft.rfft(self.Et_h) * self.normalization_factor # Add the normalization

    def backward(self):
        self.Et_v = np.fft.irfft(np.divide(self.Ef_v, self.normalization_factor)) # Add the normalization
        self.Et_h = np.fft.irfft(np.divide(self.Ef_h, self.normalization_factor)) # Add the normalization

    def apply_spectral_phase(self, spectral_phase : np.array):

        self._accumulated_phase_v += spectral_phase
        self._accumulated_phase_h += spectral_phase
        self.Ef_v = self.Ef_v*np.exp(1j*spectral_phase)
        self.Ef_h = self.Ef_h*np.exp(1j*spectral_phase)
        self.backward()

    def get_time_signal(self, units : str = 's') -> dict:


        try:
            match units.lower():
                case "s" | 'second' |'seconds':
                    unit = "s"
                    factor = 1
                case "ms" | 'milli' | 'millis' | 'millisecond' | 'milliseconds':
                    unit = "ms"
                    factor = 1E3
                case "us" | 'micro' | 'micros' | 'microsecond' | 'microseconds':
                    unit = "us"
                    factor = 1E6
                case "ns" | 'nano' | 'nanos' | 'nanosecond' | 'nanoseconds':
                    unit = "ns"
                    factor = 1E9
                case "ps" | 'pico' | 'picos' | 'picosecond' | 'picoseconds':
                    unit = "ps"
                    factor = 1E12
                case "fs" | 'femto' | 'femtos' | 'femtosecond' | 'femtoseconds':
                    unit = "fs"
                    factor = 1E15
                case "as" | 'atto' | 'attos' | 'attosecond' | 'attoseconds':
                    unit = "as"
                    factor = 1E18
        except ValueError:
            print("Invalid units given.")
        else:
            return{"units": unit, "xvals": np.multiply(self.time_axis, factor), "intensities": np.divide(self.It, factor)}

    def get_spectrum(self, units : str = 'Hz') -> dict:

        try:
            match units.lower():
                case 'hz' | 'hertz':
                    unit = "Hz"
                    xvals = self.frequency_axis
                    intensities = self.If
                case 'nm' | 'nanometers' | 'nanometres':
                    unit = "nm"
                    # convert to GHz first
                    xvals = np.divide(self.frequency_axis, 1E9)
                    # Then convert to divide. This stops run overflow caued by 1/0 gets set as max float which then has to be multiplied by 1E9
                    xvals = conv_wl_freq(np.flip(xvals))
                    # Now treat the intensities with the appropriate Jacobian
                    intensities = np.flip(self.If)*1E9 * conv_wl_freq(np.power(xvals, 2))

        except ValueError:
            print("Invalid units given.")
        else:
            return {"units" : unit, "xvals" : xvals, "intensities" : intensities}

    def remove_carrier_frequency(self):
        pass
        #self.Ef = np.sqrt(np.power(np.abs(self.Et),2 ))

    def apply_carrier_frequency(self):
        self.remove_carrier_frequency()
        #self.Et = self.Et*np.exp(1j * 2 * const.pi * self.carrier_frequency * self.time_axis)
        self.Et_v *= np.cos(2 * const.pi * self.carrier_frequency * self.time_axis)
        self.Et_h *= np.cos(2 * const.pi * self.carrier_frequency * self.time_axis)

    def apply_phase(self, taylors):
        self.remove_carrier_frequency()
        phase = np.zeros(np.shape(self.frequency_axis))
        for i, coeff in enumerate(taylors):

            phase = phase + coeff * np.divide(np.power(2*const.pi*(self.frequency_axis - self.carrier_frequency), i), scipy.special.factorial(i))

        self.apply_spectral_phase(phase)

        return phase ## delete later

    def remove_phase(self):
        self.Ef_v = self.Ef_v * np.exp(-1j * self._accumulated_phase_v)
        self.Ef_h = self.Ef_h * np.exp(-1j * self._accumulated_phase_h)
        self._accumulated_phase_v = np.zeros(np.shape(self._accumulated_phase_v))
        self._accumulated_phase_h = np.zeros(np.shape(self._accumulated_phase_h))
        self.backward()

    def t_fwhm(self, method = "interpolate"):
        if method == "interpolate":
            return find_fwhm_interpolate(self.time_axis, self.It)

    def propagate_material(self, material, length = 1E-3, anti_reflective = False):
        wavelengths = (const.c/self.frequency_axis)*1E6
        omegas = 2*const.pi*self.frequency_axis
        beta = np.divide(omegas, const.c) * material.refractive_index(wavelengths)
        # Need to introduce removing Group Delay
        phase = beta * length

        self.apply_spectral_phase(phase)


def abs2(field : np.array) -> np.array:
    return np.power(np.abs(field), 2)

def conv_wl_freq(value : float) -> float:
    return np.nan_to_num(np.divide(const.c, value), neginf=0, posinf=0)

def find_fwhm_interpolate(xs, ys):
    half_max = np.max(ys) / 2

    above_thresh = np.where(ys > half_max)[0]

    rise_xs = [xs[above_thresh[0] - 1], xs[above_thresh[0]]]
    rise_ys = [ys[above_thresh[0] - 1], ys[above_thresh[0]]]
    rise_slope = (rise_ys[1] - rise_ys[0]) / (rise_xs[1] - rise_xs[0])
    rise_intercept = rise_ys[0] - (rise_slope * rise_xs[0])

    fall_xs = [xs[above_thresh[-1] + 1], xs[above_thresh[-1]]]
    fall_ys = [ys[above_thresh[-1] + 1], ys[above_thresh[-1]]]
    fall_slope = (fall_ys[1] - fall_ys[0]) / (fall_xs[1] - fall_xs[0])
    fall_intercept = fall_ys[0] - (fall_slope * fall_xs[0])

    x_lower = (half_max - rise_intercept) / rise_slope
    x_upper = (half_max - fall_intercept) / fall_slope

    fwhm = x_upper - x_lower
    return np.abs(fwhm)

def gaussian_time(N : int, dt : float, t_fwhm : float, wavelength : float = 800E-9, pulse_energy: float = 1, polarisation = 1.0) -> RealPulse:
    pulse = RealPulse(N, dt)
    pulse.make_axes(N, dt)
    sd_t = t_fwhm/2.355
    prefactor = np.reciprocal(np.sqrt(2 * const.pi * np.power(sd_t, 2)))
    env_t = pulse_energy*prefactor*np.exp(-0.5*np.power(np.divide(pulse.time_axis, sd_t), 2))
    pulse.carrier_frequency = conv_wl_freq(wavelength)
    if isinstance(polarisation, str):
        match polarisation:
            case 'v' | 'ver' | 'vertical':
                pulse.Et_v = np.sqrt(env_t)
            case 'h' | 'hor' | 'horizontal':
                pulse.Et_h = np.sqrt(env_t)
            case _:
                raise ValueError("Invalid polarisation state.")
    elif isinstance(polarisation, Number):
        if (polarisation < 0) or (polarisation > 1):
            raise ValueError("Numerical polarisation must be equal to or between 0 and 1.")
        pulse.Et_v = np.sqrt(env_t * polarisation)
        pulse.Et_h = np.sqrt(env_t * (1- polarisation))
    else:
        raise ValueError("Invalid polarisation specification")
    
    pulse.apply_carrier_frequency()
    pulse.forward()

    return pulse

def from_spectrum(wavelength, intensities, N: int, dt: float) -> RealPulse:
    pulse = RealPulse(N, dt)
    pulse.make_axes(N, dt)

    spectrum_frequency = np.flip(conv_wl_freq(wavelength))
    intensities = np.flip(intensities) * (np.divide(const.c, np.power(spectrum_frequency, 2))) # Jacobian

    spectrum_interpolator = CubicSpline(spectrum_frequency, intensities, extrapolate=False)

    pulse.Ef = np.nan_to_num(np.sqrt(spectrum_interpolator(pulse.frequency_axis)), posinf=0, neginf=0)

    pulse.backward()

    return pulse



def attenuate(pulse: RealPulse, goggle: laser_tools.goggles.Goggle) -> RealPulse:
    attenuated_pulse = deepcopy(pulse)  # Avoid modifying the original

    valid_indices = goggle.valid_indices(attenuated_pulse.frequency_axis)

    ODs = goggle.evaluate_OD(attenuated_pulse.frequency_axis[valid_indices])
    Ef = attenuated_pulse.Ef
    Ef[valid_indices] = np.divide(Ef[valid_indices], np.power(10, ODs))
    attenuated_pulse.Ef = Ef

    return attenuated_pulse