
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from pysindy.utils import linear_damped_SHO
from pysindy.utils import cubic_damped_SHO
from pysindy.utils import linear_3D
from pysindy.utils import hopf
from pysindy.utils import lorenz
import csv

import pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Generate training data

keywords = [
    '/..../Room 1.120',
]

dt = 0.01
with open ('./whatever sensor.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile)
    selected_cols = [col for col in spamreader.fieldnames if any(k in col for k in keywords)]
    for row in spamreader: filtered_row = {col: row[col] for col in selected_cols}
    print (selected_cols)

# TODO example with real data 
