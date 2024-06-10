"""
Power Calculations
"""

import numpy as np
from scipy.integrate import quad
import pandas as pd
from scipy.interpolate import interp1d
from math import pi

def load_and_clean_data(file_path):
    """
    Load and clean the data from an Excel file.

    :param file_path: Path to the Excel file.
    :return: Cleaned DataFrame with Wavelength and Divergence columns.
    """
    #Load the data
    data = pd.read_excel(file_path)

    #Clean the data
    data.columns = ['Drop1', 'Drop2', 'Wavelength', 'Divergence']
    data = data.drop(['Drop1', 'Drop2'], axis=1)
    data = data.dropna().reset_index(drop=True)
    data['Wavelength'] = pd.to_numeric(data['Wavelength'], errors='coerce')
    data['Divergence'] = pd.to_numeric(data['Divergence'], errors='coerce')
    data = data.dropna()

    return data

def interpolate_data(data, new_wavelengths):
    """
    Interpolate the divergence data for a given set of wavelengths.

    :param data: DataFrame containing the original wavelength and divergence data.
    :param new_wavelengths: Array of wavelengths for which divergence needs to be interpolated.
    :return: Interpolated divergences for the new wavelengths.
    """
    #Extracting wavelength and divergence values from the data
    wavelengths = data['Wavelength'].values
    divergences = data['Divergence'].values

    #Creating an interpolation function
    interp_func = interp1d(wavelengths, divergences, kind='linear', fill_value='extrapolate', bounds_error=False)

    #Interpolating the divergences for the new wavelengths
    interpolated_divergences = interp_func(new_wavelengths)

    return interpolated_divergences

def calculate_beam_waist(wavelength, divergence):
    """
    Calculate the beam waist diameter using the Gaussian beam formula.

    :param wavelength: Wavelength of the laser.
    :param divergence_degrees: Divergence of the laser beam in degrees.
    :return: Beam waist.
    """
    #Calculate beam waist
    beam_waist = (720 * wavelength) / (divergence * pi**2)

    return beam_waist

file_path = r""

cleaned_data = load_and_clean_data(file_path)
new_wavelengths = [532]
interpolated_divergences = interpolate_data(cleaned_data, new_wavelengths)
wavelength = 532e-9
beam_waist = calculate_beam_waist(wavelength, interpolated_divergences[0])

#Values
P0 = 1e-3 # initial power from laser in W
w0 = beam_waist  # initial beam waist radius in meters (From collimator)
R = 0.25e-3  # radius in meters of the specular image/laser (We can choose this value)
d_f = 0.011 # diameter in meters of the fiber
z = 7.933256561   #distance in meters from source to fibers(We can change this)

def power_calc(w0, wavelength, z, R, P0=.001, d_f=0.011):
    """
    Calculate power collected at the fiber.

    Parameters
    ----------
    w0 : float
        Beam waist.
    wavelength : float
        Wavelength of the laser beam.
    z : float
        Distance from the source to the fiber.
    R : float
        Radius of the specular image.
    P0 : float, optional
        Initial power from the laser in mW. Default is 1.
    d_f : float, optional
        Diameter of the fiber in meters. Default is 0.011.

    Returns
    -------
    float
        Power collected at the fiber.

    """
    #Calculate P1
    P1 = (P0 * (d_f ** 2)) / (16 * (z ** 2))
    
    #Calculate Rayleigh range (z_R)
    z_R = (w0 ** 2 * np.pi) / wavelength
    
    #Define the Gaussian beam intensity profile
    def intensity_profile(r, z):
        w_z = w0 * ((1 + (z / z_R) ** 2) ** 0.5)
        return P1 * (1 - np.exp(-2 * (r ** 2) / (w_z ** 2)))

    #Define the integrand for power calculation
    def integrand(r, z):
        return 2 * np.pi * r * intensity_profile(r, z)

    #Perform the integration
    power_collected, _ = quad(integrand, 0, R, args=(z,))
    
    return power_collected

final_power = power_calc(beam_waist, wavelength, z, R)

print("The interpolated divergence is", interpolated_divergences)
print("Beam waist is", beam_waist, "meters")

def power_collected_collimated(w0, wavelength, z, R, P0):
    """
    calculates the power collected based on a collimated light source 

    Parameters
    ----------
    w0 : beam waist
    wavelength : wavelength of the beam
    z : distance from the source to the fiber
    R : radius of specular image

    Returns
    -------
    power_collected_coll = power collected with the collimated beam 

    """
    # Calculate Rayleigh range
    z_R = (w0 ** 2 * np.pi) / wavelength
    
    # Calculate intensity profile at the fiber
    intensity_fiber = P0 / (pi * w0**2 * (1 + (z / z_R)**2))
    
    #calculate power collected using intensity profile
    power_collected_coll = intensity_fiber * np.pi * R ** 2
    
    return power_collected_coll

power_fi = power_collected_collimated(w0, wavelength, z, R, P0)
print("Power collected using a collimated beam is", power_fi, "W")
