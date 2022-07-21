# machine_learning_altimetry_validation

The project, made available through Jupyter Notebooks, analyses the daily gridded altimetry product (Sea Level Anomalies, SLA) in the North Sea for 2004, available in SEANOE as:
Passaro Marcello, Juhl Marie-Christin (2022). Daily sea level anomalies from satellite altimetry with Random Forest Regression. SEANOE. https://doi.org/10.17882/89530 

We compare the daily machine-learning-based SLA with the latest version of the CMEMS Level 4 gridded SLA, reference number: SEALEVEL_GLO_PHY_L4_MY_008_047.

# Input needed
a) Tide Gauges data. We assembled data from GESLA-3 https://gesla787883612.wordpress.com/, correcting them for atmospheric effects (through the Dynamic Atmospher Correction) and tides. The processed data is available in the same SEANOE repository.

b) Altimetry data from Random Forest Regression. Before running the functions, the user will need to assemble the daily grids from SEANOE into one single dataframe.

c) Altimetry data from CMEMS. One single dataframe obtained assembling the CMEMS Level 4 gridded SLA (reference number: SEALEVEL_GLO_PHY_L4_MY_008_047) using the notebook "open_cmems_dataset"


## Purpose of the relevant notebooks:



1: open_cmems_dataset.ipynb: structured SLA grids from the DUACS product of CMEMS are transformed into unstructured grids of the same format of
	ds_export_forprediction. 
	
2. compare_TG_nodac.ipynb:  In this notebook, the correlation of the altimetry predicted dataset against tide gauges and CMEMS data is studied in 1D
	considering the closest point of our prediction to the tide gauge and the corresponding closest CMEMS grid point
    
3. plot_neighbour_example.ipynb: On a map, one example of the neighbouring strategy in time and space is displayed. 

4. plot_SLA_variance.ipynb: SLA variability is estimated using the interquantile range of the time series at each grid point estimated in this study and from the tide gauges. 

5. plot_dailymapsexample.ipynb: Plots of one daily grid on a geographical map, from this study and from CMEMS

6. plot_training_histograms.ipynb: Plotting functions for the histograms of the traning dataset. NOT AVAILABLE using the data in SEANOE.
