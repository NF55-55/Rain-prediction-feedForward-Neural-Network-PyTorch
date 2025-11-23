# Chance of raining predicted with a feed forward Neural Network
A simple Feedforward Neural Network (PyTorch) trained on 1950-2025 weather data (at the user-chosen location) used to predict the chance of raining based on user-provided current weather data (e.g.: [windy.com](https://www.windy.com/)).
## Brief description of the repo
* [NNmodel.pth](/NNmodel.pth): PyTorch NN model saved at the end of training.
* [NNmodelTraining.py](/NNmodelTraining.py): python code to train the feed forward NN.
* [NNtesting.py](/NNtesting.py): python code to test the NN with the testing dataset, compute the accuracy and predict the current chance of raining based on user-provided weather data.
* [inputCurrentWeatherData.csv](/inputCurrentWeatherData.csv): csv file in which the user should insert the current weather data to test/use the NN.
* [meanAndStdInput.txt](/meanAndStdInput.txt): txt file in which are saved some statistics used to normalize the input, to use during testing.
* [testData.txt](/testData.txt): testing dataset (15% of the dataset).
* [weatherData](/weatherData/): folder in which the weather data is stored.
## Weather Data
In the [weatherData](/weatherData) folder you can find the link to the documentation of the 1950-2025 hourly weather data. This dataset can be imported for any specified location from [Copernicus Climate Data Store website](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-timeseries). In my case I requested as weather variables: dewpoint temperature (2m), temperature (2m), u wind component (10m), v wind component (10m), surface pressure, skin temperature, de-accumulated total precipitation. The imported data is automatically splitted in multiple csv files, which I renamed and in this repo are mostly empty due to size limitation (the saved PyTorch NN available in this repo was actually trained on the full dataset).\
The dataset is splitted in the following way in the code: training (70%), validating (15%), testing (15%). 
