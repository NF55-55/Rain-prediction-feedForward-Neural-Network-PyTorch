import requests

from httpx import request
import torch
import torch.nn as nn

import numpy as np

# NN model
class NNmodel(nn.Module):

    def __init__(self, layer1, input_siz=6, num_classes=1):
        super(NNmodel, self).__init__()
        self.layer1 = nn.Linear(input_siz, layer1, bias=True)
        self.outpLayer = nn.Linear(layer1,num_classes, bias=True)

        self.ActFun1 = nn.ReLU()
        self.ActFun2 = nn.Sigmoid()

    def forward(self, x):
        x = self.ActFun1(self.layer1(x))
        x = self.ActFun2(self.outpLayer(x))
        
        return x

# create the NN model
model = NNmodel(30) # leaving the rest of inputs default

# import NN model
model.load_state_dict(torch.load('NNmodel.pth'))

model.eval()

model.zero_grad()

# import the metrics used to normalize the training dataset
statistics = np.loadtxt("./meanAndStdInput.txt", delimiter=",", dtype=np.float32)

statistics = torch.tensor(statistics, dtype=torch.float32)

mean = statistics[0,:]
std = statistics[1,:]

# API call to open-meteo.com to get weather data at given location
urlApi = np.loadtxt("apiUrl.txt", dtype=str)
response = requests.get(urlApi).json()
currentData = response["current"]

T2m = currentData["temperature_2m"] # temperature in °C 
print(f"Temperature: {T2m} °C")
T2m = T2m + 273.15 # converted in K

windMag = currentData["wind_speed_10m"] # magnitude of wind speed [m/s] 
print(f"Wind speed: {windMag} m/s")
windDir = currentData["wind_direction_10m"] # direction of wind in ° FROM NORTH
print(f"Wind direction: {windDir}° from N")
windDir = -(windDir-90-180)/180*np.pi # direction of wind in rad and starting from x-axis(east) and following the wind dir. convention
uWind = -windMag*np.cos(windDir) # east-west component of wind at 10 m above ground [m/s]
vWind = windMag*np.sin(windDir) # south-north component of wind at 10 m above ground [m/s]

pressure = currentData["surface_pressure"] # in hPa (windy)
pressure = pressure*100 # in Pa
print(f"Pressure: {pressure} Pa")

hourlyData = response["hourly"] # as dewT and skinT are available only in hourly
# data, then we take the prediction for the next hour as current data.
dewT = hourlyData["dew_point_2m"][0] # dew temperature in °C
print(f"Dew temperature: {dewT} °C")
dewT = dewT + 273.15 # converted in K

skinT = hourlyData["soil_temperature_0cm"][0] # in °C
print(f"Soil temperature(surface): {skinT} °C")
skinT = skinT + 273.15 # converted in K

currentData = torch.tensor((dewT, T2m, uWind, vWind, pressure, skinT), dtype=torch.float32)

# normalizing data using statistical values of training
currentNormData = (currentData - mean) / std

# output of the classification NN
outCurrent = model(currentNormData)
outCurrent = outCurrent.detach().squeeze().item()

print(f"The NN thinks there's {round(outCurrent*100,1)} % chance of raining. (75% = drizzle, 100% = rain)\n")