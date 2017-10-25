## Dataset Features
#  Calculate features of the dataset that will allow it to be compared with other datasets.
#  Peaks ~ the maximum G value in each event across all rows in each column.
#  Mean ~ the average G value in each event across all rows in each column.
#  3Sig_max/min ~ The max and min value 3 sigma for each column across rows and events.
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from scipy import log10
from scipy.signal import welch
from math import floor
from tqdm import trange


def vib_peaks (data, th=1):
    ## Find impacts over a threshhold.
    '''
    Assume data is of the form [Time, R1, R2, ... , Rn], where n is the number of response
    columns in the file.

    th ~ Threshold [G] over which peaks are recorded.
    The threshold value is only required when recording over time.  When recording impacts
    '''
    peaks = abs(data).max(axis=1).T
    mean = data.mean(axis=1).T

    return peaks, mean

def sigma_calc(data, degree):
    '''
    Calculates sigma of a set of values to a requested degree, i.e. 6 Sigma

    Input values can have as many columns as wanted.  The calculation will
    be performed across the rows.

    If a pandas Panel or 3D array is input, calculation will be performed
    across the rows and depth.

    Returns:
    [sigma1, sigma2, ..., sigman] for n columns of values input.
    '''

    mean = sp.mean(data, axis=0)
    std = sp.std(data, axis=0, ddof=1)

    for i in range(len(sp.shape(data))-2):
        mean = sp.mean(mean, axis=0)
        std = sp.std(std, axis=0, ddof=1)

    sig = [mean + degree * std, mean - degree * std]

    return(sig)

def vib_profiles(profile):
    '''
    Given a correct vibration profile, returns the profile with two
    columns.  [freq, psd]

    If profile is a .csv file, returns a previously saved file.
    '''
    vibls = pd.read_csv('../../data/vib_profiles.csv')

    input_profile = 0
    for i in range(sp.shape(vibls)[1]):
        if vibls.columns[i].upper() == profile.upper():
            input_profile = vibls.iloc[:,i:i+2].dropna()
            #input_profile = input_profile.dropna()
            input_profile.columns = ['Freq', 'PSD']
            return(input_profile)

    if profile[-4:].upper() == '.csv'.upper():
        input_profile = pd.read_csv(profile)
        return(input_profile)

    if type(input_profile) == int:
        print('Input vibration profile not found')
        input_profile = False

    return(input_profile)

def fft_data(data):
    dt = float(data.E0.Time[1] - data.E0.Time[0])    # Time Steps, [s]
    fs = float(1./dt)                                # Sampling Frequency
    N = int(len(data.E0.Time))                       # Number of Samples
    sig_len = N/fs                                   # Signal Length [s]
    df = 1/sig_len

    ## FFT, basic
    freq = df*sp.arange(0, N, dtype='d')
    data_fft = sp.fft(data.iloc[:,:,1:])
    f_fft = freq*sp.ones((len(data_fft), len(freq)))
    data_fft = sp.concatenate((f_fft[...,None], data_fft), axis=2)
    #fft_avg = sp.mean(data_fft, axis=0)

    ## PSD, basic
    f, psd = welch(data.iloc[:,:,1:], fs = fs, nperseg = N, axis=1)
    data_psd = sp.zeros((psd.shape[0], psd.shape[1], psd.shape[2]+1))
    f_psd = f*sp.ones((len(psd), len(f)))
    data_psd = sp.concatenate((f_psd[...,None], psd), axis=2)

    return(data_fft, data_psd)

def fft_dataS(data):
    tic = time.clock()

    # Maximum Time to consider data, will remove any data past
    # the last whole second
    maxT = int(floor(max(data.Time)))

    # Initialize the minimum sample length of all events
    samp_min = maxT

    # Time Step
    dt = float(data.Time[2] - data.Time[1])

    fs = float(1./dt)                                # Sampling Frequency
    N = int(len(data.Time))                       # Number of Samples
    sig_len = N/fs                                   # Signal Length [s]
    df = 1/sig_len

    ## FFT, basic
    freq = df*sp.arange(0, N, dtype='d')
    data_fft = sp.fft(data.iloc[:,1:])
    f_fft = freq*sp.ones((len(data_fft), len(freq)))
    data_fft = sp.concatenate((f_fft[...,None], data_fft), axis=2)
    #fft_avg = sp.mean(data_fft, axis=0)

    ## PSD, basic
    f, psd = welch(data.iloc[:,1:], fs = fs, nperseg = N, axis=1)
    data_psd = sp.zeros((psd.shape[0], psd.shape[1], psd.shape[2]+1))
    f_psd = f*sp.ones((len(psd), len(f)))
    data_psd = sp.concatenate((f_psd[...,None], psd), axis=2)

    return(data_fft, data_psd)

def psd_avg_data(data):

    data_fft, data_psd = fft_data(data)

    avg_psd = sp.array([[sp.mean(data_psd[:,i,j]) for j in range(len(data_psd[0,0,:]))] for i in range(len(data_psd[0,:,0]))])
    max_psd = sp.array([[max(abs(data_psd[:,i,j])) for j in range(len(data_psd[0,0,:]))] for i in range(len(data_psd[0,:,0]))])
    min_psd = sp.array([[min(abs(data_psd[:,i,j])) for j in range(len(data_psd[0,0,:]))] for i in range(len(data_psd[0,:,0]))])

    head = list(data.iloc[0].columns)
    head[0] = 'Freq'

    avg_psd = pd.DataFrame(avg_psd, columns=head)
    max_psd = pd.DataFrame(max_psd, columns=head)
    min_psd = pd.DataFrame(min_psd, columns=head)

    return(avg_psd, max_psd, min_psd)


def grms (freq, PSD):
    """Returns the Grms value for a shaped random vibration input curve.
    Input the frequency and PSD values as a list in the form grms(freq, PSD).
    The frequency and PSD list must have the same number of elements."""

    from math import log10, log

    A = 0

    if len(freq)!=len(PSD):
        print("Error: The number of elements in the Frequency and PSD lists do not match.")

    else:
        for i in range(1,len(freq)):

            # Calculate the slope
            dB = 10 * log10(PSD[i]/PSD[i-1])           # dB
            OCT = log10(freq[i]/freq[i-1])/log10(2)    # Octave
            S = dB/OCT                                 # Slope

            # Calculate the area in units of [G^2]
            if S == 0:
                A = A + PSD[i] * (freq[i] - freq[i-1])
            elif S == -3:
                A = A + -freq[i] * PSD[i] * log(freq[i-1] / freq[i])
            else:
                A = A + (3 * PSD[i]/(3 + S)) * (freq[i] - (freq[i-1]/freq[i])**(S/3) * freq[i-1])

            # Calculate the Grms [G]
            grms = A**(0.5)

    return(grms)

## Interpolate values of one profile across frequency range of another response.
def vib_trans(resp, profile):

    """
    Interpolate the values of the profile across the frequency range of the response.  The profile consists
    of two lists, a frequency and amplitude.  The response consists of the same.  This program finds the amplitudes
    of the profile at the frequencies of the response.  This allows you to compare the amplitudes of the response
    and the profile at the same frequencies.

    resp = [frequency, amplitude]
    profile = [frequency, amplitude]

    Returns the transmissibility results Respose / Input Profile.

    return([frequency, transmissibility amplitude])
    """



    # The number of axis or recordings over which to
    num_resp = min([resp.shape[1], profile.shape[1]])-1

    transo = []
    cols = ['Freq']

    for k in range(num_resp):
        m0 = []          # Finding the slope of the input profile
        for i in range(profile.shape[0]-1):
            m0.append((log10(profile.iloc[i+1,k+1])-log10(profile.iloc[i,k+1]))/(log10(profile.Freq[i+1])-log10(profile.Freq[i])))

        freq = []        # Initialize the frequency variable
        resp_c = []      # Initialize the clipped response variable
        m = []           # Initialize the slope variable
        x1 = []          # Initialize the frequency used in the point slope equation
        y1 = []          # Initialize the amplitude used in the point slope equation

        # Find the frequencies and response where which lie within the profile frequency range
        for i in trange(len(resp.Freq)):
            if resp.Freq[i] >= float(min(profile.Freq)) and resp.Freq[i] < float(max(profile.Freq)):
                freq.append(resp.Freq[i])
                resp_c.append(resp.iloc[i, k+1])
            for j in range(profile.shape[0]-1):
                if resp.Freq[i] < profile.Freq[j+1] and resp.Freq[i] >= profile.Freq[j]:
                    m.append(m0[j])
                    x1.append(profile.iloc[j+1,0])
                    y1.append(profile.iloc[j+1,k+1])

        # Make sure the slope is recording across the appropriate values.
        if len(m)!= len(freq):
            print('Error finding slope, len(m) != len(freq)')
            print('len m = %i' %len(m))
            print('len freq = %i' %len(freq))

        resp_int = []                 # Initializing the interpolated response variable.
        # Calculating the interpolated response given the slope and input profile point
        for i in range(len(freq)):
            resp_int.append(10**(m[i]*(log10(freq[i])-log10(x1[i])) + log10(y1[i])))

        # Converting the list to an array
        resp_int = sp.array(resp_int)
        resp_c = sp.array(resp_c)

        ## From Steinberg 1988
        # P_out = Q^2 * P
        # Solving for Q ->
        trans = (resp_c/resp_int)**0.5    # Q ~ Transmissibility of system

        if len(transo) == 0:
            transo = sp.array((trans), ndmin=2).T
        else:
            transo = sp.concatenate((transo, sp.array((trans), ndmin=2).T), axis=1)

        cols.append('R%i' %k)

    return(pd.DataFrame((sp.concatenate((sp.array((freq), ndmin=2).T, transo), axis=1)), columns=cols))
