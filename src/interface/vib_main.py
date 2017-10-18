## Main Program for Vibration Analysis with Pandas

import sys, os
import matplotlib.pyplot as plt

#sys.path.append(os.path.join(os.path.dirname(__file__), "../tools/"))
sys.path.append("../tools/")

from open_file_folder import *
from import_vib_data import *


#Locate the file or folder
path = get_path()
print("\n")
print("File or folder selected: \n")
print(path)
print("\n")

data = path2data(path)
