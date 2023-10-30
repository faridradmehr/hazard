import os, sys, glob
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import s3fs
import netCDF4 as nc
import zarr
from datetime import datetime
import time
import os, sys
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import xclim
import netCDF4 as nc
sys.path.append(r"/opt/app-root/src/hazard/src")
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr

class DroughtIndicator:
    def __init__(self,gcm,scenario,s3,s3_group_path):
        self.gcm = gcm     #"ACCESS-CM2",
        self.scenario = scenario
        self.s3=s3
        self.s3_group_path=s3_group_path
    def pre_chunk(self,download_dataset_flag = True, delete_existing_log_flag = False,
        years = np.arange(1950, 2101),
        variables = ['tas','pr'],
        lat_chunk_size = 40,
        lon_chunk_size = 40,
        fs_CIMP6 = s3fs.S3FileSystem(anon=True),
        datasource = NexGddpCmip6()):
        #Inner Functions
        def download_dataset(variable,year,gcm,scenario,fs=fs_CIMP6, datasource=datasource):
            scenario_ = "historical" if year < 2015 else scenario
            datapath,_ = datasource.path(gcm=gcm, scenario=scenario_, quantity=variable, year=year)
            f = fs.open(datapath, 'rb')
            ds = xr.open_dataset(f).astype('float32').compute()
            return ds

        def delete_existing_log(gcm,scenario,s3_group_path=self.s3_group_path):
            s3_rechunk_log = os.path.join(s3_group_path,'logs',"Rechunking" +"_" + gcm + "_" + scenario + ".csv")
            self.s3.rm(s3_rechunk_log)
            return
        def read_log(gcm,scenario,variable,year,years=years,variables=variables,s3_group_path=self.s3_group_path):
            s3_rechunk_log = os.path.join(s3_group_path,'logs',"Rechunking" +"_" + gcm + "_" + scenario + ".csv")
            try:
                with self.s3.open(s3_rechunk_log,'r') as f:
                    df_log = pd.read_csv(f).set_index(['Variable','Year']).astype('bool')
            except FileNotFoundError:
                df_log = (
                            pd.DataFrame([years for variable in variables],index=variables)
                            .T
                            .stack()
                            .reset_index()
                            .rename(columns={'level_1':'Variable',0:'Year'})
                            .drop(columns='level_0')
                            .assign(DownloadedFlag=False)
                            .set_index(['Variable','Year'])
                            .sort_index()
                        )
            return df_log.loc[variable,year]['DownloadedFlag']

        def update_rechunk_log(gcm,scenario,variable,year,flag=True,years=years,variables=variables,s3_group_path=self.s3_group_path):
            s3_rechunk_log = os.path.join(s3_group_path,'logs',"Rechunking" +"_" + gcm + "_" + scenario + ".csv")
            try:
                with self.s3.open(s3_rechunk_log, 'r') as f:
                    df_log = pd.read_csv(f).set_index(['Variable','Year']).astype('bool')
            except FileNotFoundError:
                df_log = (
                            pd.DataFrame([years for variable in variables],index=variables)
                            .T
                            .stack()
                            .reset_index()
                            .rename(columns={'level_1':'Variable',0:'Year'})
                            .drop(columns='level_0')
                            .assign(DownloadedFlag=False)
                            .set_index(['Variable','Year'])
                            .sort_index()
                        )
            df_log.loc[variable, year]['DownloadedFlag'] = flag
            with self.s3.open(s3_rechunk_log, 'w') as f:
                df_log.to_csv(f)
            return        

        if delete_existing_log_flag:
            delete_existing_log(self.gcm,self.scenario)
        if download_dataset_flag:
            for variable in variables:

                zarr_root = os.path.join(self.s3_group_path,variable + "_" + self.gcm + "_" + self.scenario)
                zarr_store = s3fs.S3Map(root=zarr_root,s3=self.s3,check=False)

                for year in years:
                    time_s = time.time()
                    already_processed_flag = read_log(self.gcm,self.scenario,variable,year)
                    if already_processed_flag:
                        status = "...previously processed"
                    else:
                        ds = download_dataset(variable,year,self.gcm,self.scenario).chunk({'time':365,'lat':lat_chunk_size,'lon':lon_chunk_size})
                        if year == 1950:
                            ds.to_zarr(store=zarr_store,mode='w')
                        else:
                            ds.to_zarr(store=zarr_store,append_dim='time')
                        update_rechunk_log(self.gcm,self.scenario,variable,year)
                        status = "...completed processing"
                    time_e = time.time()
                    print(f"variable = {variable}, year = {year} "  + status + "...("+f"{(time_e-time_s):.2f}sec"+")")        
    def run_single(self,run_spei_calcs_flag= True,
        delete_existing_calc_log_flag = False,
        calib_start = datetime(1985,1,1),
        calib_end = datetime(2015,1,1),
        calc_start = datetime(1985,1,1),
        calc_end = datetime(2100,12,31),
        freq = "MS",
        window = 12,
        dist = "gamma",
        method = "APP",
        lat_min = -60.0,
        lat_max = 90.0,
        lon_min = 0.0,
        lon_max = 360.0,
        lat_delta = 10.0,
        lon_delta = 10.0,
        calculate_agg_spei_data_flag = True,
        write_agg_spei_data_flag = False,
        indicator_years=[2005,2030,2040,2050,2080],
        spei_threshold=[0,-1,-1.5,-2,-2.5,-3,-3.6]
        ):
        def get_datachunks(lat_min=lat_min,lat_max=lat_max,lon_min=lon_min,lon_max=lon_max,lat_delta=lat_delta,lon_delta=lon_delta):
            lat_bins = np.arange(lat_min,lat_max + 0.1*lat_delta,lat_delta)
            lon_bins = np.arange(lon_min,lon_max + 0.1*lon_delta,lon_delta)
            data_chunks = {"Chunk_" + str(i).zfill(4) : dict(list(d[0].items())+list(d[1].items())) for i,d in enumerate(itertools.product([{'lat_min':x[0],'lat_max':x[1]} for x in zip(lat_bins[:-1],lat_bins[1:])],[{'lon_min':x[0],'lon_max':x[1]} for x in zip(lon_bins[:-1],lon_bins[1:])]))}
            return data_chunks
        data_chunks = get_datachunks()  
        
        def read_variable_from_s3_store(gcm,scenario,variable,lat_min,lat_max,lon_min,lon_max,s3_group_path=self.s3_group_path):
            zarr_root = os.path.join(s3_group_path, variable + "_" + gcm + "_" + scenario)
            zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3, check=False)
            ds = xr.open_zarr(store=zarr_store).sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
            return ds

        def calculate_spei(
                                lat_min,
                                lat_max,
                                lon_min,
                                lon_max,
                                gcm=self.gcm,
                                scenario=self.scenario,
                                s3_group_path=self.s3_group_path,
                                freq=freq,
                                window=window,
                                dist=dist,
                                method=method,
                                calib_start=calib_start,
                                calib_end=calib_end,
                                calc_start=calc_start,
                                calc_end=calc_end,
                                num_workers = 4
                            ):
            time_s0 = time.time()
            ds_tas = read_variable_from_s3_store(gcm,scenario,"tas",lat_min,lat_max,lon_min,lon_max,s3_group_path=s3_group_path).chunk({'time':100000})
            ds_pr = read_variable_from_s3_store(gcm,scenario,"pr",lat_min,lat_max,lon_min,lon_max,s3_group_path=s3_group_path).chunk({'time':100000})
            ds_tas=ds_tas.drop_duplicates(dim=...,keep='last').sortby('time')
            ds_pr=ds_pr.drop_duplicates(dim=...,keep='last').sortby('time')
            ds_pet = xclim.indices.potential_evapotranspiration(tas=ds_tas['tas'],method='MB05').astype('float32').to_dataset(name='pet')
            da_wb=xclim.indices.water_budget(pr=ds_pr['pr'],evspsblpot=ds_pet['pet'])
            with xr.set_options(keep_attrs=True):
                da_wb=da_wb-1.01*da_wb.min()
            # da_wb = da_wb #da_wb.compute() #.chunk(chunks={'time':100000})
            da_wb_calib = da_wb.sel(time=slice(str(calib_start)[:10],str(calib_end)[:10]))
            da_wb_calc = da_wb.sel(time=slice(str(calc_start)[:10],str(calc_end)[:10]))
            ds_spei = (
                            xclim.indices.standardized_precipitation_evapotranspiration_index(
                                                                                                    da_wb_calc,
                                                                                                    da_wb_calib,
                                                                                                    freq=freq,
                                                                                                    window=window,
                                                                                                    dist=dist,
                                                                                                    method=method
                                                                                                )
                            .astype('float32')
                            .to_dataset(name='spei')
                            .compute(scheduler='processes',num_workers=num_workers)
                        )
            time_e0 = time.time()
            print("(time taken: "+f"{(time_e0-time_s0):.2f}sec"+")")
            return ds_spei

        def delete_existing_calc_log(gcm,scenario,s3_group_path=self.s3_group_path):
            s3_rechunk_log = os.path.join(s3_group_path,'logs',"Calc" +"_" + gcm + "_" + scenario + ".csv")
            self.s3.rm(s3_rechunk_log)
            return

        def read_calc_log(gcm,scenario,chunk_name,s3_group_path=self.s3_group_path,data_chunks=data_chunks):
            s3_calc_log = os.path.join(s3_group_path,'logs',"Calc" +"_" + gcm + "_" + scenario + ".csv")
            try:
                with self.s3.open(s3_calc_log,'r') as f:
                    df_log = pd.read_csv(f).astype({'ChunkName':'string'}).set_index(['ChunkName']).astype('bool')
            except FileNotFoundError:
                df_log = (
                            pd.DataFrame([chunk_name for chunk_name in data_chunks],columns=['ChunkName'])
                            .astype('string')
                            .set_index('ChunkName')
                            .assign(DownloadedFlag=False)
                        )
            return df_log.loc[chunk_name]['DownloadedFlag']

        def update_calc_log(gcm,scenario,chunk_name,flag=True,s3_group_path=self.s3_group_path,data_chunks=data_chunks):
            s3_calc_log = os.path.join(s3_group_path,'logs',"Calc" +"_" + gcm + "_" + scenario + ".csv")
            try:
                with self.s3.open(s3_calc_log,'r') as f:
                    df_log = pd.read_csv(f).astype({'ChunkName':'string'}).set_index(['ChunkName']).astype('bool')
            except FileNotFoundError:
                df_log = (
                            pd.DataFrame([chunk_name for chunk_name in data_chunks],columns=['ChunkName'])
                            .astype('string')
                            .set_index('ChunkName')
                            .assign(DownloadedFlag=False)
                        )
            df_log.loc[chunk_name]['DownloadedFlag']=flag
            with self.s3.open(s3_calc_log,'w') as f:
                df_log.to_csv(f)
            return

        def get_spei_chunk_results(chunk_name,gcm,scenario,s3_group_path=self.s3_group_path):
            zarr_root = os.path.join(s3_group_path,"SPEI","ChunkedResults",gcm + "_" + scenario + "_" + chunk_name)
            zarr_store = s3fs.S3Map(root=zarr_root, s3=self.s3, check=False)
            ds = xr.open_zarr(store=zarr_store)
            return ds

        def get_spei_full_results(gcm,scenario,s3_group_path=self.s3_group_path):
            zarr_root = os.path.join(s3_group_path,"SPEI","Aggregated",gcm + "_" + scenario)
            zarr_store = s3fs.S3Map(root=zarr_root,s3=self.s3,check=False)
            ds_spei = xr.open_zarr(store=zarr_store)
            return ds_spei
        
        ### Calculating average number of months where 12 - month SPEI index is below thresholds [0,-1,-1.5,-2,-2.5,-3.6] for 20 years period        
        
        def calculate_annual_average_spei(gcm, scenario, year, spei_thr=spei_threshold, period_years=20, write_spei_indicator_flag=True, s3=self.s3, s3_group_path=self.s3_group_path):
            period=[datetime(year-period_years//2,1,1),datetime(year+period_years//2-1,12,31)]
            print(gcm+" "+scenario+" "+str(year)+" period:   "+str(period[0])+"---"+str(period[1]))
            ds_spei = get_spei_full_results(gcm, scenario)
            lats_all = ds_spei['lat'].values
            lons_all = ds_spei['lon'].values
            spei_annual = np.nan*np.zeros([len(spei_thr), len(lats_all), len(lons_all)])
            spei_temp = ds_spei.sel(time=slice(period[0], period[1]))
            lats_temp = spei_temp['lat'].values
            lons_temp = spei_temp['lon'].values
            spei_temp = spei_temp.compute()
            spei_temp = spei_temp['spei']
            for i in range(len(spei_thr)):
                spei_ext=xr.where((spei_temp <= spei_thr[i]),1,0)
                spei_ext_sum=spei_ext.mean("time")
                lats_ids = [x[0] for x in enumerate(lats_all) if x[1] in lats_temp]
                lons_ids = [x[0] for x in enumerate(lons_all) if x[1] in lons_temp]
                spei_annual[i, lats_ids[0]:lats_ids[-1]+1, lons_ids[0]:lons_ids[-1]+1] = spei_ext_sum
                spei_annual_all = xr.DataArray(spei_annual, coords={'spei_idx': spei_thr, 'lat': lats_all,'lon': lons_all,}, dims=["spei_idx","lat", "lon"])
            if write_spei_indicator_flag:
                print("Saving SPEI indicator")
                zarr_root = os.path.join(s3_group_path,"SPEI","months_spei12m_below_set",gcm + "_" + scenario+ "_" +str(year))
                zarr_store = s3fs.S3Map(root=zarr_root,s3=s3,check=False)
                target = OscZarr(prefix="hazard",store=zarr_store)
                target.write(zarr_root,spei_annual_all)
                print("SPEI Indicator calculation completed and stored at:\n"+zarr_root+"\n\n")
            return spei_annual_all            
        
        chunk_names = list(data_chunks.keys())
        num_workers = 4
    ## Calculating SPEI index
        if delete_existing_calc_log_flag:
            delete_existing_calc_log(self.gcm,self.scenario)
        if run_spei_calcs_flag:

            for chunk_name in chunk_names:

                time_s = time.time()

                data_chunk = data_chunks[chunk_name]
                lat_min_c = data_chunk['lat_min']
                lat_max_c = data_chunk['lat_max']
                lon_min_c = data_chunk['lon_min']
                lon_max_c = data_chunk['lon_max']

                already_calculated_flag = read_calc_log(self.gcm,self.scenario,chunk_name)
                if already_calculated_flag:
                    status = "... previously calculated"
                else:
                    ds_spei = calculate_spei(lat_min_c,lat_max_c,lon_min_c,lon_max_c)
                    zarr_root = os.path.join(self.s3_group_path,"SPEI","ChunkedResults",self.gcm + "_" + self.scenario + "_" + chunk_name)
                    zarr_store = s3fs.S3Map(root=zarr_root,s3=self.s3,check=False)
                    ds_spei.to_zarr(store=zarr_store,mode='w')
                    update_calc_log(self.gcm,self.scenario,chunk_name)
                    status = "... completed calculating"

                time_e = time.time()
                print(f"Chunk = {chunk_name} "  + status + " ... ("+f"{(time_e-time_s):.2f}sec"+")")
        ## Aggregating SPEI index calculations
        if calculate_agg_spei_data_flag:
            print("\n\nAggregating SPEI index calculations")
            ds_1_ = get_spei_chunk_results("Chunk_0000",self.gcm,self.scenario)
            ds_2_ = read_variable_from_s3_store(self.gcm,self.scenario,"pr",lat_min,lat_max,lon_min,lon_max)
            times_all = ds_1_['time'].values
            lats_all = ds_2_['lat'].values
            lons_all = ds_2_['lon'].values
            spei_data = np.nan*np.zeros([len(times_all), len(lats_all), len(lons_all)])

            for chunk_name in chunk_names:
                already_calculated_flag = read_calc_log(self.gcm,self.scenario,chunk_name)
                if already_calculated_flag:
                    ds_chunk = get_spei_chunk_results(chunk_name,self.gcm,self.scenario).compute()
                    spei_data_chunk = ds_chunk['spei'].values
                    lats_chunk = ds_chunk['lat'].values
                    lons_chunk = ds_chunk['lon'].values
                    lats_ids = [x[0] for x in enumerate(lats_all) if x[1] in lats_chunk]
                    lons_ids = [x[0] for x in enumerate(lons_all) if x[1] in lons_chunk]
                    spei_data[:,lats_ids[0]:lats_ids[-1]+1,lons_ids[0]:lons_ids[-1]+1] = spei_data_chunk
                print(chunk_name,end='; ')

            if write_agg_spei_data_flag:
                print("\n\Saving aggregated output")
                ds_spei_all = xr.DataArray(spei_data, coords={'time': times_all, 'lat': lats_all,'lon': lons_all,}, dims=["time","lat", "lon"]).chunk(chunks={'lat':40,'lon':40,'time':100000}).to_dataset(name='spei')
                zarr_root = os.path.join(self.s3_group_path,"SPEI","Aggregated",self.gcm + "_" + self.scenario)
                zarr_store = s3fs.S3Map(root=zarr_root,s3=self.s3,check=False)
                ds_spei_all.to_zarr(store=zarr_store,mode='w')
        #final results --- 20 years of average SPEI 
        months_spei12m_below_set= {}
        print("\n\nProducing and storing indicators ... ")
        for year in indicator_years:                
            spei_annual_average = calculate_annual_average_spei(self.gcm, self.scenario, year=year, period_years=20)
            string = '_'.join((self.gcm, self.scenario, str(year)))
            months_spei12m_below_set[string] = spei_annual_average
        self.months_spei12m_below_set= months_spei12m_below_set
        print("Completed!")