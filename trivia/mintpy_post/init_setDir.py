import sys
for p in sys.path:
    print(p)

import os
import numpy as np
import matplotlib.pyplot as plt
from mintpy import view, tsview, plot_network, plot_transection, plot_coherence_matrix

# define the work directory
work_dir = os.path.abspath(os.getcwd())

#The Path to the folder where ARIA data after running ariaTSsetup.py is stored
aria_dir = os.path.join('~/kamb-nobak/ARIA-download/Aqaba_087')

print("Work directory: ", work_dir)
print("The path to the directory with ARIA data after running ariaTSsetup:", aria_dir)

if not os.path.isdir(work_dir):
    os.makedirs(work_dir)
    print('Create directory: {}'.format(work_dir))

print('Go to work directory: {}'.format(work_dir))
os.chdir(work_dir)
