## Dataset Features
#  Calculate features of the dataset that will allow it to be compared with other datasets.
#  Peaks ~ the maximum G value in each event across all rows in each column.
#  Mean ~ the average G value in each event across all rows in each column.
#  3Sig_max/min ~ The max and min value 3 sigma for each column across rows and events.
import scipy as sp


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
    vibls = pd.read_csv('vib_profiles.csv')

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
