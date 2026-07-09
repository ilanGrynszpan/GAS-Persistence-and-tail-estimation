

# Precipitation data

Now, replicate what you did for Belo Horizonte for every other location. You should gather the locations from these folders:

Daily precipitation data is in

C:\Users\ilang\OneDrive\Documentos\Ilan\academia\dissertação\data\output

Train data is in CSV "*_train.csv"

Test data is in CSV "*_test.csv"

In case these files contain dates, you should use this same train/test partition for the whole model, as in, you should divide all other data into the same partition for model purposes.

# Covariates

- ENSO data should be in data/input/pacific/Nino34.csv, but double check if the data column represents nino34 index, I dont want incorrect data problems again.
- Dew point data is in data/input/ERA5/humidity/...
- Temperature data is in data/input/ERA5/temperature/...

Tell me before you run what you understood, from reading these files that columns are, the data column and the timestamp one, and how you have doing train/test divide.

For all input files that I listed here, check if the data column makes sense for what it should be, dont blindly run. Tell me if there are any problems.