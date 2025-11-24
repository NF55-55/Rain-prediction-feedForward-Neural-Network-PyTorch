import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

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

# test on the testing part of dataset (15% of the dataset)

testData = np.loadtxt("./testData.txt", delimiter=",", dtype=np.float32)

testData = torch.tensor(testData, dtype=torch.float32)

inpTest = testData[:, :6]
outTest = testData[:, -1].unsqueeze(1)

nCorrect = 0
outArray = []

for i in range(inpTest.shape[0]):
    out = model(inpTest[i,:])
    out = out.detach().squeeze().item()
    outArray.append(out)
    if abs(out-outTest[i,:].detach().squeeze().item())<0.5:
        nCorrect += 1

accuracy = nCorrect/outTest.shape[0] * 100

print(f"\nThe accuracy of this feedForward NN is: {round(accuracy,2)} %.\n")

# test on given data

statistics = np.loadtxt("./meanAndStdInput.txt", delimiter=",", dtype=np.float32)

statistics = torch.tensor(statistics, dtype=torch.float32)

mean = statistics[0,:]
std = statistics[1,:]

currentWeatherData = np.loadtxt("./inputCurrentWeatherData.csv", skiprows=1, delimiter=",", dtype=np.float32)

dewT = currentWeatherData[0] # dew temperature in °C (windy: https://www.windy.com/)
dewT = dewT + 273.15 # converted in K

T2m = currentWeatherData[1] # temperature in °C (windy or google)
T2m = T2m + 273.15 # converted in K

windMag = currentWeatherData[2] # magnitude of wind speed [m/s] (windy, closest weather station)
windDir = currentWeatherData[3] # direction of wind in ° FROM NORTH (windy, closest weather station)
windDir = -(windDir-90-180)/180*np.pi # direction of wind in rad and starting from x-axis(east) and following the wind dir. convention
uWind = -windMag*np.cos(windDir) # east-west component of wind at 10 m above ground [m/s]
vWind = windMag*np.sin(windDir) # south-north component of wind at 10 m above ground [m/s]
print(f"u: {uWind} m/s, v: {vWind} m/s\n")

pressure = currentWeatherData[4] # in Pa (windy)

skinT = currentWeatherData[5] # in °C (find it here https://soiltemperature.app/)
skinT = skinT + 273.15 # converted in K

currentData = torch.tensor((dewT, T2m, uWind, vWind, pressure, skinT), dtype=torch.float32)

# normalizing data using statistical values of training
currentNormData = (currentData - mean) / std

outCurrent = model(currentNormData)
outCurrent = outCurrent.detach().squeeze().item()

print(f"The NN thinks there's {round(outCurrent*100,1)} % chance of raining. (75% = drizzle, 100% = rain)\n")