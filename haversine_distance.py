import eli5
from eli5.sklearn import PermutationImportance
import pandas as pd
import xarray as xr
import glob
import os
import netCDF4
import scipy
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Distance computation
def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute distance between two lat-lon points.
    
    Parameters
    ----------
    lat1: Latitude of point 1
    lon1: Longitude of point 1
    lat2: Latitude of point 2
    lon2: Longitude of point 2
    
    Returns
    -------
    Distance
    """    
    
    
    r = 6371
    phi1 = np.radians(lat1)
    
    
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 -lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    return np.round(res, 2)