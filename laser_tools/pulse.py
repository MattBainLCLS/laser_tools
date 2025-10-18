import numpy as np
from scipy.signal import hilbert
import scipy.constants as const
import scipy.special
from scipy.interpolate import CubicSpline

import laser_tools.goggles

from copy import deepcopy


class RealPulse:

    def __init__(self, N : int, dt : float):

        self.time_axis : np.array = np.empty(shape=(1,), dtype=np.float64)
        self.frequency_axis : np.array = np.empty(shape=(1,), dtype=np.float64)
        self.Et: np.array = np.empty(shape=(1,), dtype=np.float64)
        self.Ef: np.array = np.empty(shape=(1,), dtype=np.complex128)

        self.carrier_frequency : float = None

        self.make_axes(N, dt)

        #self._accumulated_phase : np.array = np.empty(shape=(1,), dtype=np.float64)

    def make_axes(self, N : int, dt : float):
        lim = dt*(N/2)
        self.time_axis = np.arange(-lim, lim, dt)
        self.frequency_axis = np.fft.rfftfreq(N, dt)
        self.Et: np.array = np.zeros(shape=np.shape(self.time_axis), dtype=np.float64)
        self.Ef: np.array = np.zeros(shape=np.shape(self.frequency_axis), dtype=np.complex128)
        self._accumulated_phase = np.zeros(np.shape(self.frequency_axis))

        self.normalization_factor = 2*dt # Normalization factor for rfft

    @property
    def Et(self):
        return self._Et

    @Et.setter
    def Et(self, value : np.array):
        self._Et = value
        #self.forward()

    @property
    def Ef(self) -> np.array:
        return self._Ef

    @Ef.setter
    def Ef(self, value : np.array):
        self._Ef = value
        #self.backward()

    @property
    def It(self) -> np.array:
        return abs2(hilbert(self.Et))

    @It.setter
    def It(self, value : np.array):
        self.Et = np.sqrt(value)

    @property
    def If(self) -> np.array:
        return abs2(self.Ef)

    @If.setter
    def If(self, value : np.array):
        self.Ef = np.sqrt(value)

    @property
    def energy(self) -> float:
        #return np.trapezoid(self.It, x = self.time_axis)
        return np.trapezoid(self.If, x = self.frequency_axis)

    @energy.setter
    def energy(self, value : float):
        self.It = np.divide(value * self.It, self.energy)

    def forward(self):
        self.Ef = np.fft.rfft(self.Et) * self.normalization_factor # Add the normalization

    def backward(self):
        self.Et = np.fft.irfft(np.divide(self.Ef, self.normalization_factor)) # Add the normalization

    def apply_spectral_phase(self, spectral_phase : np.array):

        self._accumulated_phase += spectral_phase
        self.Ef = self.Ef*np.exp(1j*spectral_phase)
        self.backward()

    def get_time_signal(self, units = "s") -> np.array:
        return {"units" : units, "xvals" : self.time_axis, "intensities" : self.It}

    def get_spectrum(self, units = "wavelength") -> dict:

        ## Need to add jacobian
        return {"units" : units, "xvals" : conv_wl_freq(self.frequency_axis), "intensities" : self.If}
        #pass





    def remove_carrier_frequency(self):
        pass
        #self.Ef = np.sqrt(np.power(np.abs(self.Et),2 ))

    def apply_carrier_frequency(self):
        self.remove_carrier_frequency()
        #self.Et = self.Et*np.exp(1j * 2 * const.pi * self.carrier_frequency * self.time_axis)
        self.Et = self.Et * np.cos(2 * const.pi * self.carrier_frequency * self.time_axis)

    def apply_phase(self, taylors):
        self.remove_carrier_frequency()
        phase = np.zeros(np.shape(self.frequency_axis))
        for i, coeff in enumerate(taylors):

            phase = phase + coeff * np.divide(np.power(2*const.pi*(self.frequency_axis - self.carrier_frequency), i), scipy.special.factorial(i))

        self.apply_spectral_phase(phase)

        return phase ## delete later

        #phi = (taylors[1] * 2 * const.pi * self.frequencies) + (
                    #taylors[2] * (np.power(2 * const.pi * self.frequencies, 2) / scipy.special.factorial(2)))

    def remove_phase(self):
        #phase = np.arctan(np.divide(np.imag(self.Ef), np.real(self.Ef)))
        #phase = np.unwrap(np.angle(self.Ef))
        self.Ef = self.Ef * np.exp(-1j * self._accumulated_phase)
        self._accumulated_phase = np.zeros(np.shape(self._accumulated_phase))
        #self.Ef = np.sqrt(abs2(self.Ef))

        self.backward()

    def t_fwhm(self, method = "interpolate"):
        if method == "interpolate":
            return find_fwhm_interpolate(self.time_axis, self.It())

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
    return np.divide(const.c, value)

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

def gaussian_time(N : int, dt : float, t_fwhm : float, wavelength : float = 800E-9, pulse_energy: float = 1) -> RealPulse:
    pulse = RealPulse(N, dt)
    pulse.make_axes(N, dt)
    sd_t = t_fwhm/2.355
    prefactor = np.reciprocal(np.sqrt(2 * const.pi * np.power(sd_t, 2)))
    env_t = pulse_energy*prefactor*np.exp(-0.5*np.power(np.divide(pulse.time_axis, sd_t), 2))
    pulse.carrier_frequency = conv_wl_freq(wavelength)
    pulse.Et(np.sqrt(env_t))
    pulse.apply_carrier_frequency()
    pulse.forward()

    return pulse

def from_spectrum(wavelength, intensities, N: int, dt: float) -> RealPulse:
    pulse = RealPulse(N, dt)
    pulse.make_axes(N, dt)

    spectrum_frequency = np.flip(np.divide(const.c, wavelength))
    intensities = np.flip(intensities)
    intensities = intensities * (np.divide(const.c, np.power(spectrum_frequency, 2))) # Jacobian

    spectrum_interpolator = CubicSpline(spectrum_frequency, intensities, extrapolate=False)

    pulse.Ef = np.nan_to_num(np.sqrt(spectrum_interpolator(pulse.frequency_axis)))

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