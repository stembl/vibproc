## Main Program for Vibration Analysis with Pandas

import sys, os
import matplotlib.pyplot as plt

#sys.path.append(os.path.join(os.path.dirname(__file__), "../tools/"))
sys.path.append("../tools/")

from open_file_folder import *
from import_vib_data import *

## Required Information

# Input Profile
#  This can either be an industry standard or a .csv file.
input_profile_label = 'profiles/lattice_20170307-test2_logger0.csv'
#input_profile_label = 'ista air ride'

# Title of Report
title = 'Javelin Transportation 08/25/2017, Server, Logger #2'

save_title = 'javenlin_20170825-server'
save_doc_title = save_title + '.docx'

save_csv = True    # True / False
save_csv_title = save_title + '.csv'


# Locate the file or folder
path = get_path()
print("\n")
print("File or folder selected: \n")
print(path)
print("\n")

# Import and clean vibration data
data = path2data(path, eventtime = 60)

# Calculate dataset features
th = 0.5    # Threshhold, [G]
peaks, mean = vib_peaks(data, th)
sig3_max, sig3_min = sigma_calc(peaks, 3)

# Input Profile Label defined at the top
input_profile = vib_profiles(input_profile_label)
