import pandas as pd
import xarray as xr
import warnings

#from sealeveltools.sl_class import *
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from os.path import dirname
sys.path.append(dirname('/home/nemo/work_julius/vlad_globcoast/scripts/'))
sys.path.append(dirname('/home/nemo/work_julius/vlad_globcoast/scripts_vlm_mapping/'))
#from load_files import *
#import cartopy.feature as cfeature
#import cartopy.io.shapereader as shpreader

from pathlib import Path

from enum import Enum
#from pydantic import BaseModel
from typing import Union

import multiprocessing
import multiprocessing.pool
from contextlib import closing
from multiprocessing import Pool
#from nctoolbox_utils import *







def open_file(dataset,track,cycle):
    """
    
    """
    if dataset.name == 'ALES LR':
        filename = str(dataset.basedir) + '/' +'S6A_P4_2__LR_STD__NT_'+str(cycle).zfill(3)+'_'+str(track).zfill(3)+'*_F06_unvalidated.nc'
        return xr.open_dataset(glob.glob(filename)[0])

    elif dataset.name == 'EUM PDAP HR' or dataset.name == 'EUM PDAP HR':
        name_sec = 'S6A_P4_2__LR_STD__NT_' +str(cycle).zfill(3)+'_'+str(track).zfill(3)+'_*_unvalidated.nc'
        filename = str(dataset.basedir) + '/' +str(cycle).zfill(3)+'/'+name_sec
        return xr.open_dataset(glob.glob(filename)[0], group='data_20/ku')        
        
        
def get_indices(nominal_indexed,other,track,cycle):
    subindex = nominal_indexed.isel(x = (nominal_indexed['track']==track))
    lat_in = other.lat.values
    lat_nom = subindex.lat.values
    DATA = np.arange(len(lat_in))
    f=interp1d(lat_in, DATA, kind='nearest',axis=0,bounds_error=False) 
    DATA_int=f(lat_nom) #interpolate the data by nearest value
    return DATA_int,subindex



def open_nctoolbox_filename(dataset,filename):
    nc_var_mapping = dataset.nc_var_mapping
    
    file_out = get_swhdf_from_single_ncfile(filename, nc_var_mapping, nan_mask_vars=None, 
                                            external_nan_mask_file=None, 
                                      external_nan_mask_vars=None,
                                     seaice_region_to_nan=False, reduce_factor=None, out_of_range_to_nan=False, 
                                      drop_ignore_polar_regions=False,
                                     insert_dist2coast=False, drop_inland_vals=True)

    names = ['dac','dist2coast','sla','time']

    dict_ = {}
    for name in names:
        dict_[name] = (['x'],file_out[name])
    ds = xr.Dataset(dict_,
                     coords={'lon': (['x'],file_out['lon'].values ),
                            'lat': (['x'],file_out['lat'].values )}) 
    return ds

def get_data_from_track_flo(indices,data,dataset,subindex,
                        names_map = {'sla':'sla_ales','dac':'dac','qf':'qf_ales'},flag_name = 'qf_ales'):

    new_data = copy.deepcopy(subindex)
    new_data = new_data.expand_dims({'time':1})
    valid_index=~np.isnan(indices)
    indices_sel = indices[valid_index].astype(int)
    new_data = data.isel(x = indices_sel)
    time_val = np.mean(new_data.time).values
    if ' ' in str(time_val):
        time_val = pd.to_datetime(str(time_val))
    new_data = new_data.assign_coords({"time": ("time", [time_val])})
    mask_dup = list(np.diff(indices)!=0) + [False]
    new_data['sla'][:,mask_dup]=np.nan    
    return new_data


def get_filename(dataset,track,cycle):
    """
    
    """
    if dataset.name == 'ALES LR':
        filename = str(dataset.basedir) + '/' +'S6A_P4_2__LR_STD__NT_'+str(cycle).zfill(3)+'_'+str(track).zfill(3)+'*_F06_unvalidated.nc'

    elif dataset.name == 'EUM PDAP LR':
        name_sec = 'S6A_P4_2__LR_STD__NT_' +str(cycle).zfill(3)+'_'+str(track).zfill(3)+'_*_unvalidated.nc'
        filename = str(dataset.basedir) + '/' +str(cycle).zfill(3)+'/'+name_sec
        
    elif dataset.name == 'EUM PDAP HR':
        name_sec = 'S6A_P4_2__HR_STD__NT_' +str(cycle).zfill(3)+'_'+str(track).zfill(3)+'_*_unvalidated.nc'
        filename = str(dataset.basedir) + '/' +str(cycle).zfill(3)+'/'+name_sec
        
    elif dataset.name == 'CORALv2 HR':
        filename = str(dataset.basedir) + '/' +'S6A_P4_1B_HR___*'+str(cycle).zfill(3)+'_'+str(track).zfill(3)+'*.nc'
    
    elif dataset.name == 'J3-ALES LR' or dataset.name == 'J3-ADAPTIVE LR' or dataset.name == 'J3-MLE LR':
        filename = str(dataset.basedir) + '/cycle'+str(cycle).zfill(3) +'/jason3*'+str(cycle).zfill(3)+'_'+str(track).zfill(4)+'.nc'

    elif dataset.name == 'WHALES LR': 
        filename = str(dataset.basedir) +'S6A_P4_2__LR_STD__NT_'+ str(cycle).zfill(3)+'_'+str(track).zfill(3) +'*F06_unvalidated.nc'
        
    files =   glob.glob(filename)  
    if len(files) > 0:
        return files[0] 
    else:
        return None



def compute_slas(data_sub,factor = [1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],components = ['altitude','range_ales','iono_cor_gim','model_dry_tropo_cor_zero_altitude',
                 'model_wet_tropo_cor_zero_altitude','sea_state_bias_gdr','ocean_tide_sol2','ocean_tide_non_eq',
                 'internal_tide','pole_tide','dac','mean_sea_surface_sol1']):
    start = 0
    for sub_name,fac in zip(components,factor):
        start=start+data_sub[sub_name]*fac  
    data_sub['sla_ales'][:] = start
    return data_sub    
    
def get_data_from_track_flo(indices,data,dataset,subindex,filename,
                        names_map = {'sla':'sla_ales','dac':'dac','qf':'qf_ales'},flag_name = 'qf_ales'):

    new_data = copy.deepcopy(subindex)
    new_data = new_data.expand_dims({'time':1})
    valid_index=~np.isnan(indices)
    indices_sel = indices[valid_index].astype(int)
    data_sub = data.isel(x = indices_sel)
    
    if 'CORAL' in dataset.name:
        print('isin')
        sub = xr.open_dataset(filename,decode_times=False)
        start_ = pd.to_datetime(sub.time.units[18:]) 
        time_new = [start_ +pd.Timedelta(val_t,unit="ns") for val_t in data_sub.time.values]
        data_sub['time'][:]= time_new
        data_sub['time'] = data_sub['time'].astype('datetime64')    
    
    
    if len(data_sub['sla']) >0:
        time_val = np.mean(data_sub.time).values
        #print(new_data)
        if ' ' in str(time_val):
            time_val = pd.to_datetime(str(time_val))
        for varname  in list(data_sub.keys()):  
            if varname !='time':
                #print('this not',varname)
                new_data[varname] = copy.deepcopy(new_data['track'].rename(varname)*np.nan)
                #print('this',varname)
                new_data[varname][0,valid_index] = data_sub[varname].values 
        new_data = new_data.assign_coords({"time": ("time", [time_val])})
        return new_data
    else:
        return None

def retrack_outliers(DATA_in,N=20,limit=0.12,limit_cons=0.08,limit_max=2.):
    """
    median and consecutive exclusion
    
    """
    
    np.warnings.filterwarnings('ignore')
    median=pd.Series(DATA_in).rolling(window=N,center=True,min_periods=1).median().values
    #std=pd.Series(DATA_in[:,1]).rolling(window=N,center=True,min_periods=1).std().median()
    sub=DATA_in-median
    limit_max=limit_max                     # another criterium exclude all greater/smaller 4m
    DATA_in[abs(sub)>limit]=np.nan      # exclude all greater than median
    DATA_in[abs(sub)>limit_max]=np.nan  # and more than 2m in total
    
    # consecutive difference 3 cases:
    # high diff and nan    (1) then exclude the next
    # nan and high diff    (2) case ok test for 1 and 2
    # no nan and high diff (3) then exclude the one which has the largest distance to the next or previous point
    # two points surrounded by nans exclude both
    
    
    c_diff=np.append(np.diff(DATA_in),[np.nan,np.nan,np.nan])
    ind_out=(abs(c_diff)>limit_cons).nonzero()[0]
    ind_out=ind_out[ind_out>1]
    if len(ind_out)>0:   
        out=~((abs(c_diff[ind_out+1])>abs(c_diff[ind_out-1])) | np.isnan(c_diff[ind_out+1])) # skip these 
        
        out_nan=np.isnan(c_diff[ind_out-1])  # set to nan if before is nan
        out_nan_2nd=np.isnan(c_diff[ind_out+2]) # set +1 and +2 to nan if +3 is nan
        ind_nan2nd=np.append(ind_out[out_nan_2nd]+1,ind_out[out_nan_2nd]+2)
        
        ind_nan=np.append(ind_out[out],ind_out[~out]+1)
        ind_nan=np.append(ind_nan,ind_out[out_nan])
        ind_nan=np.append(ind_nan,ind_nan2nd)
        
        ind_nan=ind_nan[(0<=ind_nan) & (ind_nan<len(sub))]
        DATA_in[ind_nan]=np.nan
    return DATA_in#,std

def kick_outliers(series,thresh=3):
    series=series.where(abs(series)<2.)
    series['1997':'1997-07']=series['1997':'1997-07'].where(abs(series['1997':'1997-07'])<0.5)
    series=series.where(abs(series)<series.std()*thresh)
    return series


def set_s6_settings(version = 'v3'):
    """
    version v1 : 30 september 2022
    version v2 : 28 mar 2023
    
    """
    s6_settings = {}
    s6_settings['base_dir'] = '/home/oelsmann/data/projects/S6_JTEX/'
    s6_settings['processing_names'] = ['EUM PDAP HR','EUM PDAP LR','ALES LR','CORALv2 HR','J3-ALES LR','J3-ADAPTIVE LR','J3-MLE LR','WHALES']
    s6_settings['data_sets'] = [eum_pdap_hr,eum_pdap_lr,ales_lr,coral_hr,j3_ales_lr,j3_adaptive_lr,j3_mle_lr,whales_lr]
    
    s6_settings['directories'] = ['']
    s6_settings['cycles'] = [[188,197]]
    s6_settings['time_series_dir'] = '/nfs/DGFI8/H/work_julius/S6/time_series/'
    s6_settings['version'] = version
    
    s6_settings['ranges_dist2coast'] = [[0,5],[5,10],[10,15],[15,20],[0,10],[10,20],[20,30],[30,40],[40,50]]
    s6_settings['ranges_dist2_tg'] = [[0,5],[5,10],[10,15],[15,20],[0,10],[10,20],[20,30],[30,40],
                                       [40,50],[50,60],[60,70],[70,80],[90,100],[150,160]]    
    return s6_settings

def mkdr(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")    
        
def read_cycle_to_time_series(pool_input_list):
    cycle,dataset,track,names_map,flag_name,nominal_indexed =pool_input_list
    data = open_file(dataset,track,cycle)
    print('cycle:',cycle)
    indices,subindex = get_indices(nominal_indexed,data,track,cycle)

    data_mapped = get_data_from_track(indices,data,dataset,subindex,
                    names_map = names_map,flag_name = flag_name)
    return data_mapped

def compute_S6_time_series(dataset,dirout,cycles,tracks,nominal_indexed,
                           flag_name,names_map,start_pool,pool_workers):
    """
    
    """

    s6_settings = set_s6_settings()
    for track in tracks:
        print('track: ',track)
        if start_pool:
            pool_input =[]
            for cycle in cycles:
                pool_input.append([cycle,dataset,track,names_map,flag_name,nominal_indexed])

            with closing(Pool(processes=pool_workers)) as pool:    
                out = pool.map(read_cycle_to_time_series,pool_input)
                pool.terminate()  
        else: 
            out = []
            for cycle in cycles:
                data = open_file(dataset,track,cycle)
                indices,subindex = get_indices(nominal_indexed,data,track,cycle)
                data_mapped = get_data_from_track(indices,data,dataset,subindex,
                                names_map = names_map,flag_name = flag_name)
                print('cycle:',cycle)
                out.append(data_mapped)
        data_stacked = xr.concat(out,dim='time')
        data_stacked.to_netcdf(dirout + str(track).zfill(3)+'_'+str(cycles[0])+'_'+str(cycles[-1])+'_'+dataset.name.replace(" ", "_")+'_'+s6_settings['version']+'.nc')
        print(str(track) + ' saved!')

        
        
def compute_S6_time_series_for_datasets_toolboxsync(x=0,testing=False,start_pool = True,
                                                    pool_workers=10,add='update'):
    """loop through datasets and compute time series


    """
    
    s6_settings = set_s6_settings()
    if testing:
        start_pool = False
    for dataset in s6_settings['data_sets'][x:x+1]:
        print(dataset.name)
        basedir = str(dataset.basedir)

        names_map = {'sla':'sla_ales','dac':'dac'}
        flag_name = 'qf_ales'

        nc_var_mapping=dataset.nc_var_mapping
        if dataset.name in ['EUM PDAP HR']:
            nc_var_mapping['distance_to_coast']='dist2coast'

        nominal_indexed = xr.open_dataset('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_indexed'+add+'.nc')
        tracks = pd.DataFrame(nominal_indexed['track'].values).drop_duplicates().values.astype(int).flatten()
        cycles_range = dataset.options['cycles']
        cycles = np.linspace(cycles_range[0],cycles_range[1],cycles_range[1]-cycles_range[0]+1).astype(int)

        dirout = s6_settings['time_series_dir']+dataset.name.replace(" ", "_")+'/'
        mkdr(dirout)

        for track in tracks:
            print('track: ',track)
            if start_pool:
                pool_input =[]
                for cycle in cycles:
                    pool_input.append([cycle,dataset,track,names_map,flag_name,nominal_indexed])

                with closing(Pool(processes=pool_workers)) as pool:    
                    out = pool.map(read_cycle_to_time_series_nctoolbox,pool_input)
                    pool.terminate()  
            else: 
                out = []
                for cycle in cycles:
                    print('cycle:',cycle)
                    filename = get_filename(dataset,track,cycle)
                    if filename is not None:
                        data = open_nctoolbox_filename(dataset,filename)
                        #data = open_file(dataset,track,cycle)
                        indices,subindex = get_indices(nominal_indexed,data,track,cycle)
                        data_mapped = get_data_from_track_flo(indices,data,dataset,subindex,filename)
                        #print('cycle:',cycle)
                        out.append(data_mapped)
            if not testing:
                out = [i for i in out if i is not None]
                if len(out) > 0:
                    data_stacked = xr.concat(out,dim='time')
                    data_stacked.to_netcdf(dirout + str(track).zfill(3)+'_'+str(cycles[0])+'_'+str(cycles[-1])+'_'+dataset.name.replace(" ", "_")+'_'+s6_settings['version']+'.nc')
                    print(str(track) + ' saved!')  
                            
        
def compute_S6_time_series_for_datasets(start_pool = True,pool_workers=10):
    """loop through datasets and compute time series


    """

    s6_settings = set_s6_settings()
    for dataset in s6_settings['data_sets'][2:3]:
        basedir = str(dataset.basedir)
        add_ = dataset.options['add_']

        names_map = {'sla':'sla_ales','dac':'dac'}
        flag_name = 'qf_ales'
        nominal_indexed = xr.open_dataset('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_indexed.nc')
        tracks = pd.DataFrame(nominal_indexed['track'].values).drop_duplicates().values.astype(int).flatten()
        cycles_range = dataset.options['cycles']
        cycles = np.linspace(cycles_range[0],cycles_range[1],cycles_range[1]-cycles_range[0]+1).astype(int)

        dirout = s6_settings['time_series_dir']+dataset.name.replace(" ", "_")+'/'
        mkdr(dirout)

        compute_S6_time_series(dataset,dirout,cycles,tracks,nominal_indexed,
                                   flag_name,names_map,start_pool,pool_workers)
        
### GESLA        

class GeslaDataset:
    """A class for loading data from GESLA text files into convenient in-memory
    data objects. By default, single file requests are loaded into
    `pandas.DataFrame` objects, which are similar to in-memory spreadsheets.
    Multifile requests are loaded into `xarray.Dataset` objects, which are
    similar to in-memory NetCDF files."""

    def __init__(self, meta_file, data_path):
        """Initialize loading data from a GESLA database.

        Args:
            meta_file (string): path to the metadata file in .csv format.
            data_path (string): path to the directory containing GESLA data
                files.
        """
        self.meta = pd.read_csv(meta_file)
        self.meta.columns = [
            c.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .lower()
            for c in self.meta.columns
        ]
        self.meta.loc[:, "start_date_time"] = [
            pd.to_datetime(d) for d in self.meta.loc[:, "start_date_time"]
        ]
        self.meta.loc[:, "end_date_time"] = [
            pd.to_datetime(d) for d in self.meta.loc[:, "end_date_time"]
        ]
        self.meta.rename(columns={"file_name": "filename"}, inplace=True)
        self.data_path = data_path

            
    def file_to_pandas(self,filename, return_meta=True,apply_use_flag=False,
                       start_date =None, resampling = None):
        """Read a GESLA data file into a pandas.DataFrame object. Metadata is
        returned as a pandas.Series object.

        Args:
            filename (string): name of the GESLA data file. Do not prepend path.
            return_meta (bool, optional): determines if metadata is returned as
                a second function output. Defaults to True.

        Returns:
            pandas.DataFrame: sea-level values and flags with datetime index.
            pandas.Series: record metadata. This return can be excluded by
                setting return_meta=False.
        """
        with open(self.data_path + filename, "r") as f:
            data = pd.read_csv(
                f,
                skiprows=41,
                names=["date", "time", "sea_level", "qc_flag", "use_flag"],
                sep="\s+",
                parse_dates=[[0, 1]],
                index_col=0,
            )
            duplicates = data.index.duplicated()
            if duplicates.sum() > 0:
                data = data.loc[~duplicates]
                warnings.warn(
                    "Duplicate timestamps in file " + filename + " were removed.",
                )

            if apply_use_flag:
                data = data[data['use_flag']==1]
            if resampling != None:
                data = data.resample(resampling).mean()
            if start_date != None:
                data = data[data.index > start_date]

            if return_meta:
                meta = self.meta.loc[self.meta.filename == filename].iloc[0]
                return data, meta
            else:
                return data

    def files_to_xarray(self, filenames,apply_use_flag=False,
                       start_date =None, resampling = None):

        """Read a list of GESLA filenames into a xarray.Dataset object. The
        dataset includes variables containing metadata for each record.

        Args:
            filenames (list): list of filename strings.

        Returns:
            xarray.Dataset: data, flags, and metadata for each record.
        """
        files = []
        act_filenames = []
        
        for f in filenames:
            outfile = self.file_to_pandas(f, return_meta=False,apply_use_flag=apply_use_flag,
                                              start_date =start_date, resampling = resampling)
            if len(outfile) > 1:
                print(f)
                files.append(outfile.to_xarray())
                act_filenames.append(f)
                
        if len(files) > 0:
            data = xr.concat(files,dim="station")

            #idx = [s.Index for s in self.meta.itertuples() if s.filename in act_filenames]
            idx = []
            for fname in act_filenames:
                idx.append(self.meta[self.meta['filename']==fname].index[0])            
            meta = self.meta.loc[idx]
            meta.index = range(meta.index.size)
            meta.index.name = "station"
            data = data.assign({c: meta[c] for c in meta.columns})
            return data            
            

    def load_N_closest(self, lat, lon, N=1, force_xarray=False):
        """Load the N closest GESLA records to a lat/lon location into a
        xarray.Dataset object. The dataset includes variables containing
        metadata for each record.

        Args:
            lat (float): latitude on the interval [-90, 90]
            lon (float): longitude on the interval [-180, 180]
            N (int, optional): number of locations to load. Defaults to 1.
            force_xarray (bool, optional): if N=1, the default behavior is to
                return a pandas.DataFrame object containing data/flags and a
                pandas.Series object containing metadata. Set this argument to
                True to return a xarray Dataset even if N=1. Defaults to False.

        Returns:
            xarray.Dataset: data, flags, and metadata for each record.
        """
        N = int(N)
        if N <= 0:
            raise Exception("Must specify N > 0")

        d = (self.meta.longitude - lon) ** 2 + (self.meta.latitude - lat) ** 2
        idx = d.sort_values().iloc[:N].index
        meta = self.meta.loc[idx]

        if (N > 1) or force_xarray:
            return self.files_to_xarray(meta.filename.tolist())

        else:
            data, meta = self.file_to_pandas(meta.filename.values[0])
            return data, meta

    def load_lat_lon_range(
        self,
        south_lat=-90,
        north_lat=90,
        west_lon=-180,
        east_lon=180,
        force_xarray=False,
    ):
        """Load GESLA records within a rectangular lat/lon range into a xarray.
        Dataset object.

        Args:
            south_lat (float, optional): southern extent of the range. Defaults
                to -90.
            north_lat (float, optional): northern extent of the range. Defaults
                to 90.
            west_lon (float, optional): western extent of the range. Defaults
                to -180.
            east_lon (float, optional): eastern extent of the range. Defaults
                to 180.
            force_xarray (bool, optional): if there is only one record in the
                lat/lon range, the default behavior is to return a
                pandas.DataFrame object containing data/flags and a
                pandas.Series object containing metadata. Set this argument to
                True to return a xarray.Dataset even if only one record is
                selected. Defaults to False.

        Returns:
            xarray.Dataset: data, flags, and metadata for each record.
        """
        if west_lon > 0 & east_lon < 0:
            lon_bool = (self.meta.longitude >= west_lon) | (
                self.meta.longitude <= east_lon
            )
        else:
            lon_bool = (self.meta.longitude >= west_lon) & (
                self.meta.longitude <= east_lon
            )
        lat_bool = (self.meta.latitude >= south_lat) & (self.meta.latitude <= north_lat)
        meta = self.meta.loc[lon_bool & lat_bool]

        if (meta.index.size > 1) or force_xarray:
            return self.files_to_xarray(meta.filename.tolist())

        else:
            data, meta = self.file_to_pandas(meta.filename.values[0])
            return data, meta

def min_lon(phi,min_km):
    lonmin=round(min_km/(((math.cos(phi*(2*3.14/360))*6371)*2)*3.14/360),1)
    return lonmin 

def make_coords(tg,min_km,min_lat,loc=False):
    if loc:
        TG_loc=[None]*2
        TG_loc[0]=tg[0]
        TG_loc[1]=tg[1]
        lat_up=TG_loc[0]+min_lat
        lat_down=TG_loc[0]-min_lat     
    else:
        TG_loc=[None]*2
        TG_loc[0]=tg[0]
        TG_loc[1]=tg[1]
        lat_up=TG_loc[0]+min_lat
        lat_down=TG_loc[0]-min_lat 
    
    if -180+min_lon(TG_loc[0],min_km) < TG_loc[1] < 180-min_lon(TG_loc[0],min_km):
        lon_up=TG_loc[1]+min_lon(TG_loc[0],min_km)
        lon_down=TG_loc[1]-min_lon(TG_loc[0],min_km)
    elif -180+min_lon(TG_loc[0],min_km) > TG_loc[1]:
        lon_up=TG_loc[1]+min_lon(TG_loc[0],min_km)
        lon_down=360+TG_loc[1]-min_lon(TG_loc[0],min_km)   
    else:
        lon_up=TG_loc[1]+min_lon(TG_loc[0],min_km)-360
        lon_down=TG_loc[1]-min_lon(TG_loc[0],min_km)
    # more concrete to 0 - 360
    if (lon_down<0) & (lon_up<=0):
        lon_up=lon_up+360
        lon_down=lon_down+360
    elif (lon_down > 0) & (lon_up < 0):
        lon_up = lon_up + 360
    return [lat_up,lat_down,lon_up,lon_down]

def make_indices(add='update'):
    nominal_track = xr.open_dataset('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz.nc')
    gesla = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess'+add+'.nc')
    all_coords =[]
    min_km = 250
    tgs_ = np.vstack([gesla.latitude.values,gesla.longitude.values]).T
    min_lat = min_km/111
    for i in range(len(tgs_[:,0])):
        all_coords.append(make_coords(tgs_[i,:],min_km,min_lat,loc=False))
    indices_all = []
    indices = np.arange(len(nominal_track.x))

    lons = nominal_track.lon.values
    lats = nominal_track.lat.values

    for coord in all_coords:
        indices_all.append(indices[(lons > coord[3]) & (lons < coord[2]) &  (lats > coord[1]) & (lats < coord[0])])    
    vals_ = pd.DataFrame(indices_all)
    vals_f = vals_.values.flatten()
    indices_final =pd.DataFrame(vals_f).dropna().drop_duplicates().values.flatten().astype(int)
    nominal_track['track'][np.sort(indices_final)].to_dataset().to_netcdf('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_indexed'+add+'.nc')



    
#def haversine(coord1, coord2):
#    R = 6372800  # Earth radius in meters
#    lat1, lon1 = coord1
#    lat2, lon2 = coord2
#    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
#    dphi       = np.radians(lat2 - lat1)
#    dlambda    = np.radians(lon2 - lon1)
#    a = np.sin(dphi/2)**2 + \
#        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
#    if str(type(a))=="<class 'numpy.float64'>":
#        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
#    else:
#        return 2*R*np.arctan(np.sqrt(a), np.sqrt(1 - a))
  
    
    
def get_dist2coast(add='update'):
    #gesla = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess.nc')
    nominal_indexed = xr.open_dataset('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_indexed'+add+'.nc')
    s6_settings = set_s6_settings()
    dataset = s6_settings['data_sets'][0]
    dirout = s6_settings['time_series_dir']+dataset.name.replace(" ", "_")+'/'
    tracks = pd.DataFrame(nominal_indexed['track'].values).drop_duplicates().values.astype(int).flatten()
    cycles_range = dataset.options['cycles']
    cycles = np.linspace(cycles_range[0],cycles_range[1],cycles_range[1]-cycles_range[0]+1).astype(int)
    dists = []
    for track in tracks:
        file = xr.open_dataset(dirout  + str(track).zfill(3)+'_13_22_EUM_PDAP_HR_'+s6_settings['version']+'.nc')
        dists.append(file['dist2coast'].mean(dim='time').values)
    nominal_indexed['dist2coast'] = copy.deepcopy(nominal_indexed['track'].rename('dist2coast')*np.nan)
    nominal_indexed['dist2coast'][:] = np.concatenate(dists)
    nominal_indexed.to_netcdf('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_dist2coast_indexed'+add+'.nc')    
    
def make_gesla(start_date = pd.to_datetime('01-01-2021'),add='update'):
    dire = '/nfs/DGFI8/D/tide-gauges/GESLA_3/'
    #filenames = os.listdir(dire)
    meta = '/nfs/DGFI8/D/tide-gauges/GESLA_3_meta/GESLA3_ALL_2.csv'    
    gesla = GeslaDataset(meta,dire)
    #filenames = gesla.meta[gesla.meta['end_date_time'] > pd.to_datetime('31-12-2021')].filename.values
    filenames = gesla.meta.filename.values
    gesla_all = gesla.files_to_xarray(filenames = filenames,apply_use_flag=True,
                       start_date =start_date, resampling = 'H')
    gesla_all.to_netcdf('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021'+add+'.nc')

    
def select_and_dropdupl(add='update_northsea4'):
    gesla_all = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021'+add+'.nc')
    gesla_all_sorted = gesla_all.sortby('site_name')
    limit = 1500
    counts = gesla_all_sorted['sea_level'].count(dim='date_time').values
    coords =np.stack([gesla_all_sorted.latitude.values,gesla_all_sorted.longitude.values])
    distancs = []
    for coord1 in coords.T:
        distancs.append(haversine(coord1, coords))
    distancs = np.vstack(distancs)
    pairs = []
    reject_indices = []
    for i in range(distancs.shape[0]):
        sub = distancs[:,i]
        idx = np.argwhere(   (sub<limit) & (sub!=0) ) 
        if len(idx)>0:
            pairs.append([i,idx[0]])
            if counts[i]<=counts[int(idx[0])]:
                reject_indices.append(i)
            else:
                reject_indices.append(int(idx[0]))
    reject_at  = pd.DataFrame(reject_indices).drop_duplicates().values.flatten()
    gesla_selected = gesla_all_sorted.sel({'station':~np.isin(np.arange(distancs.shape[0]),reject_at)})
    mask = (gesla_selected['date_time'].to_dataframe()['date_time'] >  pd.to_datetime('04-15-2021')).values
    counts_after_april = gesla_selected.sel({'date_time':mask}).count(dim='date_time')['sea_level']
    gesla_selected_final = gesla_selected.where(counts_after_april > 1,drop=True)
    gesla_selected_final.attrs={'comment':'duplicates are rejected; longest TG record of a TG pair within 1.5km is kept. TG must contain data after 04-15-2021. Use-flag applied. Hourly averaged.'}
    gesla_selected_final.to_netcdf('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX'+add+'.nc')
    
def compute_dac_gesla(add='update'):
    gesla = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess'+add+'.nc')
    dataset= gesla
    dataset = dataset.rename({'latitude':'lat','longitude':'lon'})
    lon= dataset['lon']
    lon[lon <0 ] = lon[lon <0 ]+360
    dataset['lon'][:] = lon
    make_DAC_aviso(dataset)
def make_DAC_aviso(dataset,DAC_name='',name = 'dac_dif_26296_12.nc',dire = '/DGFI8/D/ib/DAC.AVISO/',years_in = [2021],out_dir='/nfs/public_ads/Oelsmann/marcello/gesla_v3/'):
    testfile = xr.open_dataset(dire +str(years_in[0])+'/'+ name)
    testfile=testfile.rename({'latitude':'lat','longitude':'lon'})
    lon_nw,lat_nw = np.meshgrid(testfile.lon,testfile.lat)
    start = np.full(lon_nw.shape,False)
    id1=[]
    id2=[]
    for lat,lon in zip(dataset.lat.values,dataset.lon.values):
        diff =abs((lon_nw-lon))+ abs((lat_nw-lat))
        out = np.where(diff == np.min(diff))
        id1.append(out[0][0])
        id2.append(out[1][0])
    for year in years_in:
        data  = glob.glob(dire+str(year)+'/*nc')
        DATA=np.empty((len(data),len(id1)))*np.nan
        TIME=pd.DataFrame(np.empty(len(data))*np.nan)
        i=0
        for dataname in data:
            print(i, dataname)
            dat = xr.open_dataset(dataname)

            DATA[i,:]=dat.dac.values[id1,id2]
            TIME.loc[i]=np.datetime64(dat.dac.date[:-4])     
            i=i+1
        DATA = pd.DataFrame(DATA)
        DATA['time']=TIME
        DATA.to_csv(out_dir+'data_for_time_'+str(year))
        
def correct_gesla_for_dac():    
    dire = '/DGFI8/D/ib/DAC.AVISO/'
    years_in = [2021]
    out_dir='/nfs/public_ads/Oelsmann/marcello/gesla_v3/'
    add='update_northsea'
    gesla = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess'+add+'.nc')
    dac = pd.read_csv(out_dir+'data_for_time_'+str(years_in[0])).sort_values(by='time')
    dac=dac.set_index('time')
    dac.index = pd.to_datetime(dac.index)
    dac_sub = dac[(pd.DatetimeIndex(dac.index.values) >= pd.Timestamp(gesla['date_time'][0].values)) & (pd.DatetimeIndex(dac.index.values) <= pd.Timestamp(gesla['date_time'][-1].values))]
    gesla_time = gesla.loc[dict(date_time=slice(dac.index[0], dac.index[-1]))]
    gesla_time['dac'] = copy.deepcopy(gesla_time['sea_level_lowess'])*np.nan
    dac_res = (dac_sub.resample('1H').mean()).interpolate(method='cubic',limit=10)
    gesla_time['dac'][:,np.isin(gesla_time.date_time,dac_res.index)] = dac_res.iloc[:,1:].values.T
    gesla_time['sla_dac']=gesla_time['sea_level_lowess']-gesla_time['dac']
    gesla_time['sla_dac_no_loess']=gesla_time['sea_level']-gesla_time['dac']
    gesla_time.to_netcdf('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess_dac_corrected'+add+'.nc')

    
### compute correlations    
    
def sel_vals_by_lon_lat(coord1,data,min_km = 250):
    indices = np.arange(len(data.x))
    lons = data.lon.values
    lats = data.lat.values    
    min_lat = min_km/111
    all_coords=make_coords(coord1,min_km,min_lat,loc=False)    
    indices_all = indices[(lons > all_coords[3]) & (lons < all_coords[2]) &  (lats > all_coords[1]) & (lats < all_coords[0])]
    return data.isel(x = indices_all)

def select_tracks(nominal_track,gesla,station = 0):
    print('Select tracks for station ',station)
    i = station
    tgs_ = np.vstack([gesla.latitude.values,gesla.longitude.values]).T
    coord1 = tgs_[i,:]
    dat_out = sel_vals_by_lon_lat(coord1,nominal_track,min_km = 250)
    tracks_sel = pd.DataFrame(dat_out['track'].values).drop_duplicates().values.flatten().astype(int)
    return tracks_sel,coord1

def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    if str(type(a))=="<class 'numpy.float64'>":
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    else:
        return 2*R*np.arctan(np.sqrt(a), np.sqrt(1 - a))
    
    
def compute_dist_dist_tg(files,nominal_indexed,coord1,tracks_sel,min_km = 250,files_add=None):
    print('compute distance + distance to tg')
    vals_concat = xr.concat(files,dim='x')
    if files_add is not None:
        print('is in2')
        vals_correct = xr.concat(files_add,dim='x')
        vals_concat = vals_concat.assign_coords({"time": vals_correct.time.values})
    vals_concat['dist2coast'] = nominal_indexed.isel(x = np.isin(nominal_indexed['track'].values,tracks_sel))['dist2coast']
    for name_ in ['lon','lat']:
        if 'time' in vals_concat[name_].dims:
            vals_concat[name_] = vals_concat[name_].mean(dim='time')    

    vals_concat_sel = sel_vals_by_lon_lat(coord1,vals_concat,min_km = min_km)
    vals_concat_sel = vals_concat_sel.where(vals_concat_sel['dist2coast']>0,drop=True)

    dist = haversine(coord1, [vals_concat_sel.lat.values,vals_concat_sel.lon.values])
    vals_concat_sel['dist2tg'] = copy.deepcopy(vals_concat_sel['dist2coast'].rename('dist2tg'))*np.nan
    vals_concat_sel['dist2tg'][:] = dist
    return vals_concat_sel

# outlier check
def outlier_check(data,name = 'sla',max_sl=1.5):
    data = data.where(abs(data[name]).max(dim='time') <max_sl,drop=True)
    return data

def compute_correlation(vals_concat_sel_not_nan,gesla_tg,tracks_sel,
                        tolerance=pd.Timedelta('5H'),ranges_dist2coast=[[0,5]],
                        ranges_dist2_tg=[[0,5]]):
    # correlation for every single track again
    corr_arrays= []
    for track in tracks_sel:
        vals_concat_sel_not_nan_sub = vals_concat_sel_not_nan.isel(x=  (vals_concat_sel_not_nan['track'] == track)).dropna(dim='time',how='all')
        if len(vals_concat_sel_not_nan_sub.time) > 1 and len(vals_concat_sel_not_nan_sub.x) > 14:
            df = pd.DataFrame([2]*len(vals_concat_sel_not_nan_sub.time),index=pd.to_datetime(vals_concat_sel_not_nan_sub.time.values))
            gesla_tg_df = gesla_tg[['sea_level_lowess','sla_dac']].to_dataframe()
            merged_gesla =pd.merge_asof(df,gesla_tg_df, 
                                left_index=True, 
                                right_index=True, 
                                direction='nearest', 
                                tolerance=tolerance)
            
            merged_gesla['dac'] = vals_concat_sel_not_nan_sub.where(vals_concat_sel_not_nan_sub['dist2tg'] ==vals_concat_sel_not_nan_sub['dist2tg'].min(),drop=True)['dac'].values
            merged_gesla['sea_level_lowess_minus_dac'] = merged_gesla['sea_level_lowess'] - merged_gesla['dac']
            corr_track = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sea_level_lowess_minus_dac'])
            corr_track_without_dac = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values-vals_concat_sel_not_nan_sub['dac'].values ,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sea_level_lowess'])
            corr_track_self_corr = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sla_dac'])
            
            data_out,exist_1 = select_data_sub(vals_concat_sel_not_nan_sub,selector_name = 'dist2coast',
                                       ranges = np.asarray(ranges_dist2coast)*1000.,min_len=2)
            data_out_2,exist_2 = select_data_sub(vals_concat_sel_not_nan_sub,selector_name = 'dist2tg',
                                       ranges = np.asarray(ranges_dist2_tg)*1000.,min_len=2)        

            corrs_ = []
            for data in [data_out,data_out_2]:
                corrs_.append(pd.DataFrame(data['sla'].values,index = data.time.values).corrwith(merged_gesla['sla_dac']))        

                        
            
            count_ = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).mul(merged_gesla['sea_level_lowess_minus_dac'],axis=0).count().values

            dummy = vals_concat_sel_not_nan_sub.mean(dim='time')
            dummy['sla']=dummy['sla']*np.nan
            dummy=dummy.rename({'sla':'correlation'})
            dummy['correlation'][:] = corr_track.values
            dummy['correlation (no dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation (no dac)'][:] = corr_track_without_dac.values     
            dummy['correlation (self dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation (self dac)'][:] = corr_track_self_corr.values     
            dummy['correlation per dist2coast (self dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation per dist2coast (self dac)'][:len(corrs_[0])] = corrs_[0].values 

            dummy['correlation per dist2tg (self dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation per dist2tg (self dac)'][:len(corrs_[1])] = corrs_[1].values  


            for subname,subdata in zip(['dist2coast','dist2tg'],[data_out,data_out_2]):
                for name in ['dist2coast','dist2tg','track']:    
                    dummy[subname+'_'+name]= copy.deepcopy(dummy['correlation']*np.nan)
                    dummy[subname+'_'+name][:len(subdata[name])] = subdata[name].values

            dummy['count'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['count'][:] = count_  
            corr_arrays.append(dummy)
            
    if len(corr_arrays) > 0:
        all_corrs = xr.concat(corr_arrays,dim='x')
        return all_corrs
    
def select_data_sub(datain,selector_name = '',ranges = [[0,5]],min_len=5):
    data_out = []
    exist_ = []
    for range_sub in ranges:

        selector = (range_sub[0] <= datain[selector_name]) & (datain[selector_name] <= range_sub[1] )
        data_comb = datain.isel({'x':selector})

        if len(data_comb.x) >= min_len:
            data_out.append(data_comb.isel({'x':[0]})*0 + data_comb.mean(dim='x'))
            exist_.append(True)
        else:
            data_out.append(datain.isel({'x':[0]})*np.nan)
            exist_.append(False)
    return xr.concat(data_out,dim='x'),exist_


def compute_correlation_old(vals_concat_sel_not_nan,gesla_tg,tracks_sel,tolerance=pd.Timedelta('5H')):
    # correlation for every single track again
    corr_arrays= []
    for track in tracks_sel:
        vals_concat_sel_not_nan_sub = vals_concat_sel_not_nan.isel(x=  (vals_concat_sel_not_nan['track'] == track)).dropna(dim='time',how='all')
        if len(vals_concat_sel_not_nan_sub.time) > 1:
            df = pd.DataFrame([2]*len(vals_concat_sel_not_nan_sub.time),index=pd.to_datetime(vals_concat_sel_not_nan_sub.time.values))
            gesla_tg_df = gesla_tg[['sea_level_lowess','sla_dac']].to_dataframe()
            merged_gesla =pd.merge_asof(df,gesla_tg_df, 
                                left_index=True, 
                                right_index=True, 
                                direction='nearest', 
                                tolerance=tolerance)
            
            merged_gesla['dac'] = vals_concat_sel_not_nan_sub.where(vals_concat_sel_not_nan_sub['dist2tg'] ==vals_concat_sel_not_nan_sub['dist2tg'].min(),drop=True)['dac'].values
            merged_gesla['sea_level_lowess_minus_dac'] = merged_gesla['sea_level_lowess'] - merged_gesla['dac']
            corr_track = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sea_level_lowess_minus_dac'])
            corr_track_without_dac = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values-vals_concat_sel_not_nan_sub['dac'].values ,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sea_level_lowess'])
            corr_track_self_corr = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).corrwith(merged_gesla['sla_dac'])
            
            
                        
            
            count_ = pd.DataFrame(vals_concat_sel_not_nan_sub['sla'].values,index = vals_concat_sel_not_nan_sub.time.values).mul(merged_gesla['sea_level_lowess_minus_dac'],axis=0).count().values

            dummy = vals_concat_sel_not_nan_sub.mean(dim='time')
            dummy['sla']=dummy['sla']*np.nan
            dummy=dummy.rename({'sla':'correlation'})
            dummy['correlation'][:] = corr_track.values
            dummy['correlation (no dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation (no dac)'][:] = corr_track_without_dac.values     
            dummy['correlation (self dac)'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['correlation (self dac)'][:] = corr_track_self_corr.values     
            dummy['count'] = copy.deepcopy(dummy['correlation']*np.nan)
            dummy['count'][:] = count_  
            corr_arrays.append(dummy)
            
    if len(corr_arrays) > 0:
        all_corrs = xr.concat(corr_arrays,dim='x')
        return all_corrs

def compute_correlation_over(data_in):
    nominal_indexed,station,dirout,cycles,dataset,s6_settings,gesla = data_in
    tracks_sel,coord1 = select_tracks(nominal_indexed,gesla,station = station)
    files = []
    for track in tracks_sel:
        print(track)
        name = dirout + str(track).zfill(3)+'_'+str(cycles[0])+'_'+str(cycles[-1])+'_'+dataset.name.replace(" ", "_")+'_'+s6_settings['version']+'.nc'
        out = xr.open_dataset(name)
        if 'J3' in  dataset.name:
            time_new = [pd.to_datetime('01-01-1985') +pd.Timedelta(val_t,unit="s") for val_t in out.time.values]
            out = out.assign_coords({"time": time_new})        
        out['track'][:] = track
        out['track'] =out['track'].mean(dim='time')
        files.append(out.dropna(dim='time',how='all'))
    files_add = None
    if 'J3ffffff' in dataset.name:
        # old function
        print('is in')
        # get time from S6 dataset
        files_add = []
        dirout2 = s6_settings['time_series_dir']+s6_settings['data_sets'][0].name.replace(" ", "_")+'/'
        cycles_range2 = s6_settings['data_sets'][0].options['cycles']
        cycles2 = np.linspace(cycles_range2[0],cycles_range2[1],cycles_range2[1]-cycles_range2[0]+1).astype(int)
    
        for track in tracks_sel:
            print(track)
            name = dirout2 + str(track).zfill(3)+'_'+str(cycles2[0])+'_'+str(cycles2[-1])+'_'+s6_settings['data_sets'][0].name.replace(" ", "_")+'_'+s6_settings['version']+'.nc'
            out = xr.open_dataset(name)
            out['track'][:] = track
            out['track'] =out['track'].mean(dim='time')
            files_add.append(out.dropna(dim='time',how='all'))        

    if len(files) > 0 :
        vals_concat_sel = compute_dist_dist_tg(files,nominal_indexed,coord1,tracks_sel,min_km = 250,files_add=files_add)

        # match data
        min_samples = 5
        tolerance = '5H' # hours
        gesla_tg = gesla.isel(station = station)

        vals_concat_sel_new = vals_concat_sel.where(vals_concat_sel['sla'].count(dim='time')>=min_samples,drop=True)
        vals_concat_sel_not_nan = vals_concat_sel_new.isel(x = ~np.isnan(vals_concat_sel_new['sla'].mean(dim='time')))
        vals_concat_sel_not_nan = outlier_check(vals_concat_sel_not_nan,name = 'sla',max_sl=1.5)
        all_corrs = compute_correlation(vals_concat_sel_not_nan,gesla_tg,tracks_sel,
                                        tolerance=pd.Timedelta('5H'),
                                        ranges_dist2coast=s6_settings['ranges_dist2coast'],ranges_dist2_tg=s6_settings['ranges_dist2_tg']) 
        
        
        if all_corrs is not None:
            all_corrs['station'] = copy.deepcopy(all_corrs['track'])
            all_corrs['station'][:] = station   
            return all_corrs    

        
def compute_correlations(testing=True,pool_workers = 10,start_pool = True,x= 2,add = 'update'):
    """ compute correlation between gesla and j3,j6 data
    with/without dac correction

    """
    gesla = xr.open_dataset('/nfs/public_ads/Oelsmann/marcello/gesla_v3/gesla_2021_selected_JTEX_lowess_dac_corrected'+add+'.nc')
    s6_settings = set_s6_settings()
    

    
    
    if testing:
        start_pool = False
    for dataset in s6_settings['data_sets'][x:x+1]:
        print(dataset.name)
        basedir = str(dataset.basedir)
        names_map = {'sla':'sla_ales','dac':'dac'}
        flag_name = 'qf_ales'
        nc_var_mapping=dataset.nc_var_mapping
        if dataset.name in ['EUM PDAP HR']:
            nc_var_mapping['distance_to_coast']='dist2coast'
        nominal_indexed = xr.open_dataset('/home/nemo/work_julius/S6_JTEX/data/S6_20Hz_dist2coast_indexed.nc')
        # use old nominal indexed
        tracks = pd.DataFrame(nominal_indexed['track'].values).drop_duplicates().values.astype(int).flatten()
        cycles_range = dataset.options['cycles']
        cycles = np.linspace(cycles_range[0],cycles_range[1],cycles_range[1]-cycles_range[0]+1).astype(int)
        dirout = s6_settings['time_series_dir']+dataset.name.replace(" ", "_")+'/'
        corr_concat = []
        if start_pool:
            pool_input =[]
            for station in np.arange(len(gesla.station)):
                pool_input.append([nominal_indexed,station,dirout,cycles,dataset,s6_settings,gesla])
            with closing(Pool(processes=pool_workers)) as pool:    
                corr_concat = pool.map(compute_correlation_over,pool_input)
                pool.terminate()           

        else:
            for station in np.arange(len(gesla.station))[:1]:
                print(station)
                all_corrs = compute_correlation_over([nominal_indexed,station,dirout,cycles,dataset,s6_settings,gesla])
                if all_corrs is not None:
                    corr_concat.append(all_corrs)
        out = [i for i in corr_concat if i is not None]
        out = xr.concat(out,dim='x')
    if not testing:
        print('save data!')
        store_dir = '/nfs/public_ads/Oelsmann/marcello/gesla_v3/correlations/'
        out.to_netcdf(store_dir+'Correlations_'+dataset.name.replace(" ", "_")+'_vs_gesla_'+s6_settings['version']+'.nc')
        
        
        out_frame = out.to_dataframe()
        ranges_ = (np.linspace(0,50,6)*1000)
        out_frame['group dist2coast'] = np.empty(len(out_frame))*np.nan
        out_frame['group dist2tg'] = np.empty(len(out_frame))*np.nan

        out_frame['stations dist2coast'] = np.empty(len(out_frame))*np.nan
        out_frame['stations dist2tg'] = np.empty(len(out_frame))*np.nan

        for i in range(len(ranges_)-1):
            for name in ['dist2coast','dist2tg']:
                mask1=(out_frame[name]  >= ranges_[i]) & (out_frame[name]  <= ranges_[i+1])
                out_frame['group '+name][mask1]=str(ranges_[i])+' =< x < '+ str(ranges_[i+1])
                out_frame['stations '+name][mask1]= out_frame['station'][mask1].drop_duplicates().count()
        
        out_frame.to_csv(store_dir+'Correlations_'+dataset.name.replace(" ", "_")+'_vs_gesla_table'+s6_settings['version']+'.csv')

#compute_correlations(testing=False,pool_workers = 10,start_pool = True,x= 1,add = 'update')                
        
def compute_all_results():
    make_gesla(start_date = pd.to_datetime('01-01-2021'),add='update')
    select_and_dropdupl(add='update')
    compute_dac_gesla(add='update')
    correct_gesla_for_dac()
    
    
    