from MesoPy import Meso
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import requests
import os
import warnings

def retr_wxobs_synopticlabs(api_key,data_path,station_id='knyc',
                            st_time='201801010000',ed_time='201801020000',download_new=False):
    """Function to retrieve timeseries weather observations from an observation site. Uses the 
    MesoWest/SynopticLabs API to retrieve the observations. 
    
    **********
    PARAMETERS
        api_key: SynopticLabs api_key. 
        data_path: Path to directory in which we want to save the observations. 
        station_id: Four-letter station id. 
        st_time: start time for observations, in format 'YYYYMMDDhhmm'
        ed_time: end time for observations, in format 'YYYYMMDDhhmm'
        
    OUTPUT: 
        obs_dict: Dictionary with station attributes, station observations, observation units, and 
                  quality control summary. 
        path_name: File path/name for data that was just retrieved. 
    """
        
    def get_synopticlabs_token(api_key):
        request_generate_token = 'http://api.mesowest.net/v2/auth?apikey='+api_key
        api_out = requests.get(request_generate_token).text
        token_dict = json.loads(api_out)
        token_synopticlabs = token_dict['TOKEN']
        return token_synopticlabs
    
    def get_station_attrs(data_ts,station_attrs): 
        station_info = {}
        for attr in station_attrs: 
            station_info[attr] = data_ts['STATION'][0][attr]
        return station_info
    
    def get_station_obs(data_ts,vbl_list): 
        station_data = {}
        station_data[vbl_list[0]] = data_ts['STATION'][0]['OBSERVATIONS'][vbl_list[0]]
        for vbl in vbl_list[1:]:
            station_data[vbl] = data_ts['STATION'][0]['OBSERVATIONS'][
                                        list(data_ts['STATION'][0]['SENSOR_VARIABLES'][vbl].keys())[0]]
        return station_data
    
    vbl_list = ['date_time','air_temp','relative_humidity']
    station_attrs = ['STID','ELEVATION','NAME','LONGITUDE','LATITUDE']
    file_name = 'wxobs_'+station_id+'_'+st_time+'_'+ed_time+'.npy'
    
    token_synopticlabs = get_synopticlabs_token(api_key)
    
    if download_new: 
        # Retrieve the station data from API.
        m = Meso(token=token_synopticlabs)
        data_ts = m.timeseries(stid=station_id, start=st_time, end=ed_time, vars=vbl_list[1:])    

        # Put everything in a single dictionary. 
        obs_dict = {}
        obs_dict['station_attrs'] = get_station_attrs(data_ts,station_attrs)
        obs_dict['station_obs'] = get_station_obs(data_ts,vbl_list)
        obs_dict['units'] = data_ts['UNITS']
        obs_dict['qc_summary'] = data_ts['QC_SUMMARY']

        # Save the dictionary
        np.save(os.path.join('.','data',file_name),obs_dict) 
        
    else: 

        # Load the dictionary
        try: 
            obs_dict = np.load(os.path.join('.','data',file_name)).item()
        except: 
            raise(OSError('File not found: '+filename))
    
    return obs_dict, os.path.join(data_path,file_name)

def obs_dict2df(wx_obs): 
    """Place observations in dictionary returned from 'retr_wxobs_synopticlabs' in a dataframe."""
    obs_df = pd.DataFrame(wx_obs['station_obs'])
    obs_df['date_time'] = pd.to_datetime(obs_df['date_time'],utc=True).dt.tz_convert('EST')
    obs_df = obs_df.set_index('date_time')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_df = obs_df.resample('H').agg('nearest') # Resample to nearest hour. 
    return obs_df

def add_time_feats(obs_df): 
    """Add time features to observations dataframe"""
    obs_df['hour'] = obs_df.index.hour
    obs_df['day_of_week'] = obs_df.index.dayofweek
    return obs_df

def correct_for_climate_change(obs_df,slope):
    corr_obs_df = obs_df.copy()
    obs_df['air_temp'] = obs_df['air_temp']+slope*(2020-obs_df['year'])
    return corr_obs_df

def plot_load_estimations(obs_df): 
    """Plot time series of temperature, relative humidity, and load estimations."""
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(311)
    ax.plot(obs_df.index.to_list(),obs_df['air_temp'],color='r')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (C)')
    ax.set_xlim([min(obs_df.index.to_list()),max(obs_df.index.to_list())])
    ax.grid(True)

    ax = fig.add_subplot(312)
    ax.plot(obs_df.index.to_list(),obs_df['relative_humidity'],color='g')
    ax.set_xlabel('Date')
    ax.set_ylabel('Relative Humidity (%)')
    ax.set_xlim([min(obs_df.index.to_list()),max(obs_df.index.to_list())])
    ax.grid(True)

    ax = fig.add_subplot(313)
    ax.plot(obs_df.index.to_list(),obs_df['load'],color='k')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity load (MWh)')
    ax.set_ylim([0,10000])
    ax.set_xlim([min(obs_df.index.to_list()),max(obs_df.index.to_list())])
    ax.grid(True)
    
    plt.tight_layout()
    fig.savefig(os.path.join('.','figs','load_estimator_plot.png'))
    
    return fig

def total_load_by_day(obs_df): 
    """Calculate total electricity demand (load) for each day in the timeseries"""
    obs_df['date'] = obs_df.index.date
    daily_load = obs_df.groupby('date')[['load']].agg('sum')
    daily_load['date'] = pd.to_datetime(daily_load.index.to_series())
    daily_load = daily_load.set_index('date')
    return daily_load

def get_load_dist(daily_load,center_day,window_rad,weekday=True): 
    """Compiles electricity load data for date ranges centered on 'center_day', for both 
    weekends and weekdays. Does not account for leap years.
    """
    if weekday: 
        daily_load = daily_load[daily_load['dayofweek'].isin([1,2,3,4,5])]
    else: 
        daily_load = daily_load[daily_load['dayofweek'].isin([0,6])]
    if center_day-window_rad<0: 
        sub = daily_load[(daily_load['dayofyear']>=(center_day-window_rad)%365) | 
                         (daily_load['dayofyear']<=(center_day+window_rad)%365)]
    elif center_day+window_rad>364: 
        sub = daily_load[(daily_load['dayofyear']>=(center_day-window_rad)%365) | 
                         (daily_load['dayofyear']<=(center_day+window_rad)%365)]
    else: 
        sub = daily_load[(daily_load['dayofyear']>=(center_day-window_rad)%365) & 
                         (daily_load['dayofyear']<=(center_day+window_rad)%365)]
    return sub

def calc_daily_load_distributions(daily_load,window_rad=7): 
    """Compiles median, 5th, 95th, 1st, and 99th percentile electricity loads using a sliding window"""
    
    daily_load['dayofyear'] = daily_load.index.dayofyear
    daily_load['dayofweek'] = daily_load.index.dayofweek
    
    # For weekdays
    load_forecasts_week = pd.DataFrame({})
    for center_day in range(1,366): 
        load_dist = get_load_dist(daily_load,center_day,window_rad,weekday=True)
        load_forecasts_week[center_day] = load_dist['load'].quantile(q=[0.01,0.05,0.5,0.95,0.99])
        
    # For weekends  
    load_forecasts_wknd = pd.DataFrame({})
    for center_day in range(1,366): 
        load_dist = get_load_dist(daily_load,center_day,window_rad,weekday=False)
        load_forecasts_wknd[center_day] = load_dist['load'].quantile(q=[0.01,0.05,0.5,0.95,0.99])
        
    return load_forecasts_week.transpose(),load_forecasts_wknd.transpose()

def date_to_dayofyear(date='20120101'): 
    """Gets day of year for a given date string"""
    return pd.Timestamp(date).dayofyear

def plot_forecasts(load_forecasts):
    """Plots energy forecasts"""
    date_list = [pd.Timestamp(year=2020,month=1,day=1)+pd.Timedelta('1 days')*i for i in range(1,366)]
    fig = plt.figure(figsize=[10,4])
    ax = fig.add_subplot(111)
    ax.plot(date_list,load_forecasts[0.5],'r-')
    ax.fill_between(date_list,load_forecasts[0.01],load_forecasts[0.99],color='grey',alpha=0.25)
    ax.fill_between(date_list,load_forecasts[0.05],load_forecasts[0.95],color='grey',alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('NYC daily electricity demand (MWh)')
    ax.grid(True)
    ax.set_ylim([60000,175000])
    ax.set_xlim([min(date_list),max(date_list)])
    return fig