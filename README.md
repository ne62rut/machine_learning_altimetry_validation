# machine_learning_altimetry_validation

The project analyses results obtained with the algorithm of machine_learning_altimetry

## Validation and data analysis:



1:  open_cmems_dataset.ipynb: structured SLA grids from the DUACS product of CMEMS are transformed into unstructured grids of the same format of
	ds_export_forprediction. Note the possibility to load the CMEMS solution considering all satellites, vs the possibility
	to use the two_sat solution

2. open_DMI_model.ipynb In this routine, the Baltic Model Reanalysis, available as structured grids, are transformed into unstructured grids of the same format of
	ds_export_forprediction. Moreover, the sea level anomalies are corrected by finding the corresponding interpolated Dynamic Atmosphere Correction.
	
3.  compare_TG_nodac.ipynb:  In this routine, the correlation of the altimetry predicted dataset against tide gauges and CMEMS data is studied in 1D
	considering the closest point of our prediction to the tide gauge and the corresponding closest CMEMS grid point
    
4. plot_neighbour_example.ipynb: On a map, one example of the neighbouring strategy in time and space is displayed. Note that
the code is a simple copy-paste of parts of create_ML_training and its functions, so it is not automatically updated in case of changes to the program
