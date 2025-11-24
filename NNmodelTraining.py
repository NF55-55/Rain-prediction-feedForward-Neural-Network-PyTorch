import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

data = np.loadtxt("./weatherData/dewTemp and 2mTemp.csv", delimiter=",", skiprows=1,
                   usecols=(1,2), dtype=np.float32)

dewT = data[:,0] # dew temp. at 2 m above ground
T2m = data[:,1] # temp. 2 m above ground
# converted into torch array
dewT = torch.tensor(dewT, dtype=torch.float32)
T2m = torch.tensor(T2m, dtype=torch.float32)

data = np.loadtxt("./weatherData/east-north wind components at 10m.csv", delimiter=",", skiprows=1,
                   usecols=(1,2), dtype=np.float32)

uWind = data[:,0] # east-west component of wind at 10 m above ground 
vWind = data[:,1] # south-north component of wind at 10 m above ground
# converted into torch array
uWind = torch.tensor(uWind, dtype=torch.float32)
vWind = torch.tensor(vWind, dtype=torch.float32)

data = np.loadtxt("./weatherData/pressure and precipitation.csv", delimiter=",", skiprows=1,
                  usecols=(1,2), dtype=np.float32)

pressure = data[:,0] # pressure
hourlyPrecm = data[:,1] # decumulated precipitation per hour in m
# converted into torch array
pressure = torch.tensor(pressure, dtype=torch.float32)
hourlyPrecm = torch.tensor(hourlyPrecm, dtype=torch.float32)

skinT = np.loadtxt("./weatherData/surface temperature.csv", delimiter=",", skiprows=1,
                  usecols=(1), dtype=np.float32) # temperature of the surface (ground or sea)
# converted into torch array
skinT = torch.tensor(skinT, dtype=torch.float32)

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

#Create our loss function
criterion = nn.BCELoss()

#Define our learning rate and optimizer
lr = 1e-2
optimizer = optim.AdamW(model.parameters(), lr=lr)

# joining together the inputs to then split into training (70%), validation (15%) and test (15%) datasets
dataset = torch.cat((dewT.unsqueeze(1), T2m.unsqueeze(1), uWind.unsqueeze(1), vWind.unsqueeze(1), pressure.unsqueeze(1),
                    skinT.unsqueeze(1)), dim=1)

# normalizing the inputs so that they have the same weight on the output
print(f"Mean of dataset: {dataset.mean(dim=0)}\n")
print(f"Standard dev. of dataset: {dataset.std(dim=0)}\n")
with open("./meanAndStdInput.txt", "w") as f:
    f.write(f"{dataset.mean(dim=0)[0]},{dataset.mean(dim=0)[1]},{dataset.mean(dim=0)[2]},{dataset.mean(dim=0)[3]},"
            f"{dataset.mean(dim=0)[4]},{dataset.mean(dim=0)[5]}\n")
    f.write(f"{dataset.std(dim=0)[0]},{dataset.std(dim=0)[1]},{dataset.std(dim=0)[2]},{dataset.std(dim=0)[3]},"
            f"{dataset.std(dim=0)[4]},{dataset.std(dim=0)[5]}")
dataset = (dataset - dataset.mean(dim=0)) / dataset.std(dim=0)

print(dataset.shape) 

expOut = torch.zeros(hourlyPrecm.shape[0], dtype=torch.float32) # expected output (0=no rain, 1=rain) initialized to "no rain"

for i in range(hourlyPrecm.shape[0]):
    if hourlyPrecm[i] >= 0.0005: # 0.0005 m or 0.5 mm of rain per hour is the lower bound of "light rain" definition
        expOut[i] = 1
    if hourlyPrecm[i] >= 0.0001 and hourlyPrecm[i] < 0.0005: # between 0.1 mm/h and 0.5 mm/h still consider it slight rain (classif. = 0.8)
        expOut[i] = 0.75

# now add the expected output as last column of the dataset, so to shuffle it together
dataset = torch.cat((dataset, expOut.unsqueeze(1)), dim=1)

print(dataset.shape)

# shuffle randomly the dataset (no time dependency in the training, train with many different years data)
indices = torch.randperm(dataset.size(0))
dataset = dataset[indices] # shuffled dataset

# divide dataset into training, validation and test dataset
nRows70perc = round(dataset.shape[0]*0.70)
nRows85perc = round(dataset.shape[0]*0.85)
inpTrain = dataset[:nRows70perc, :6]
outTrain = dataset[:nRows70perc, -1].unsqueeze(1)
print(outTrain.shape)
lenTrain = inpTrain.shape[0]

inpValid = dataset[nRows70perc:nRows85perc, :6]
outValid = dataset[nRows70perc:nRows85perc, -1].unsqueeze(1)
lenValid = inpValid.shape[0]

with open("testData.txt", "w") as f:
    for i in range(dataset[nRows85perc+1:-1, 0].shape[0]):
        f.write(f"{dataset[nRows85perc+1+i,0]},{dataset[nRows85perc+1+i,1]},{dataset[nRows85perc+1+i,2]},{dataset[nRows85perc+1+i,3]},"
                f"{dataset[nRows85perc+1+i,4]},{dataset[nRows85perc+1+i,5]},{dataset[nRows85perc+1+i,6]}\n")

# training function to use at each epoch
def train_epoch(model, inpTrain, outTrain, criterion, optimizer, loss_logger, inpValid, outValid, loss_log_validation):

    # Forward pass of model
    output = model(inpTrain)

    # Calculate loss
    loss = criterion(output, outTrain)

    optimizer.zero_grad()

    # Backprop loss
    loss.backward()

    # Optimization Step
    optimizer.step()

    loss_logger.append(loss.item())

    # validation step (to check if there's overfitting
    out_validation = model(inpValid)
    loss_validation = criterion(out_validation, outValid)
    loss_log_validation.append(loss_validation.item())

    return loss_logger, loss_log_validation

## Training
train_loss = []
loss_log_validation = []
nEpochs = 450

model.train()

for i in trange(nEpochs, desc="Epoch", leave=False):
    train_loss, loss_log_validation = train_epoch(model, inpTrain, outTrain, criterion, optimizer, train_loss,
                                                                inpValid, outValid, loss_log_validation)

# saving model
torch.save(model.state_dict(), 'NNmodel.pth') # saving the NN model (for later testing)
print("\n------------------------Model saved!-------------------------\n")

# plot of function loss per each epoch
plt.figure()
plt.plot(np.array(range(len(train_loss))), train_loss, 'b-*')
plt.plot(np.array(range(len(loss_log_validation))), loss_log_validation, 'r-*')
plt.grid(True)
plt.xlabel("# of epochs")
plt.ylabel("Loss function per epoch")
plt.legend(["Training", "Validation"])
plt.show()