#!/usr/bin/env python
import numpy as np
from scipy.signal import butter, lfilter, resample, filtfilt, iirnotch
from scipy import stats

def detect_peaks(ecg_measurements,signal_frequency,gain = 1000):
        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25

        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)


        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)
        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)
        squared_ecg_measurements = differentiated_ecg_measurements ** 2
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices

 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order = 1):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def notch_filter(data, notch, sample_Fs = 500):
    quality_factor = 30.0
    for notch_freq in notch:
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, sample_Fs)
        data = filtfilt(b_notch, a_notch, data)
    return data

def findpeaks(data, spacing=1, limit=None):
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind

