# Fitbit_SurgicalPrediction
Code for IMWUT/Ubicomp 2022 Predicting Post-Operative Complications with Wearables: A Case Study with Patients Undergoing Pancreatic Surgery

## Run the code
Preprocess Fitbit data
```
python extract_preop.py
```

Impute short missing segments in heart rate time series data. Imputation threshold is set to 10 minutes here. 
```
python raw_impute_threshold.py
```

Compute daily features (with standardization). Extraction threshold is set to 8 hours here.
```
python feature_generate.py.py
```

Extract high-level features with SSA. A vector is generated for each user. 
```
python generate_input_ssa.py
```
