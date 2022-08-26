# -*- coding: utf-8 -*-
"""El Nino LSTM Model (temp and wwv anom).py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import datetime
import julian

import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter, detrend
from scipy import stats


#manually set seeds (helps with reproducibility)
torch.manual_seed(0)
np.random.seed(0)


################################################################################
                            #Notebook Goal#
# Create an LSTM model, train it on a certain amount of years of the El Nino 3.4
# forecasts temperature and wwv anomalies, and have it predict the remaining 
#years of the data set. Then compare these results to the known data.
################################################################################



################################################################################
                          # GLOBAL VARIABLES #

#input length/ Number of days used to predict
xWindow = 14

#Number of days to average over for prediction
yWindow = 7

#Number of leading days
leadDays = 15
#Number of years used for training, rest for testing
#testingBegin = 240 corresponds to testing beginning in Jan 2000. So, first
# prediction would be xWindow months after that.
testingBegin = 34

#parameters for smoothing, the window length and polynomal order
w, p = 25, 1


#now we create our hyperparameter variables

num_epochs = 800 #1000 epochs
learning_rate = 0.001 #0.001 lr
################################################################################


"""
Transforms long sequence of data into a series of shorter sequences,
which will act as a sliding scale of data values. The intended training inputs
will be arrays of length xWindow, and the intended outputs will be subsequent
arrays of length Window.

Input: sequence of testing data (dataframe), number months after input
window that we want to predict (int)(e.g. mosAfter = 2: predict month that's
two months after input window)

Output: a tensor with dimensions (len(arr)-xWindow)*xWindow and a tensor with 
dimensions (len(arr)-xWindow)*1

All leap days are removed for easy shaping
"""

#open daily data and create np array of DJF days
file = open('erl114015_Detrended_Electricity_demand_GB.dat')
data = np.array([["Data", "Detrended_demand_GWh"]])
for line in file:
    x = line.split()
    month = x[1][6:8]
    if month == '12' or month == '01' or month == '02':
        data = np.vstack((data, [x[1][1:-1], x[2][1:-1]]))

elec = data[1:,1].astype(int)

#open nao predictor data and create 2d np array of winter DJF years
nao = np.genfromtxt("norm.daily.nao.index.b500101.current.ascii", dtype=float)[:,3]
nao = nao[9184:23054].reshape((-1, 365))

#RMM1 and RMM2 tsv, open and reshape in 2d DJF years
RMM1 = pd.read_csv('RMM1.tsv', sep='\t')
RMM2 = pd.read_csv('RMM2.tsv', sep='\t')

m1 = np.array(RMM1['RMM1'][273:14143]).reshape((-1, 365))
m2 = np.array(RMM2['RMM2'][273:14143]).reshape((-1, 365))

#some NaN values are present, replace NaN date with average of that date across all other years
nao_nan = np.nanmean(nao, axis = 0)
nao = detrend(np.where(np.isnan(nao), nao_nan, nao).flatten()).reshape((-1, 365))
m1_nan = np.nanmean(m1, axis = 0)
m1 = detrend(np.where(np.isnan(m1), m1_nan, m1).flatten()).reshape((-1, 365))
m2_nan = np.nanmean(m2, axis = 0)
m2 = detrend(np.where(np.isnan(m2), m2_nan, m2).flatten()).reshape((-1, 365))


def slidingScaleY(data, y_window):
    y_arr = np.array([])
    data = data.reshape((-1, 90))
    
    for i in range(len(data)):
        for j in range(len(data[0])-y_window+1):
            y_arr = np.append(y_arr, np.mean(data[i][j:j+y_window]))
    y_var = Variable(torch.from_numpy(y_arr).float())
    return y_var

#train and test sets, normalized by mu and std of train set
y_train_tensors = slidingScaleY(elec[:testingBegin*90], yWindow)
mu_y, std_y = torch.mean(y_train_tensors), torch.std(y_train_tensors)
y_train_tensors = (y_train_tensors - mu_y)/std_y
y_test_tensors = slidingScaleY(elec[testingBegin*90:], yWindow)
y_test_tensors = (y_test_tensors - mu_y)/std_y

y_train_tensors = y_train_tensors.reshape(y_train_tensors.shape[0], 1)

def slidingScaleX(data, x_window, y_window, lead_days):
    x_arr = np.zeros((1, x_window))
    
    for i in range(len(data)):
        for j in range(275-x_window-lead_days+1, 365-y_window+1-x_window-lead_days+1):
            x_arr = np.vstack((x_arr, data[i][j:j+x_window]))
    x_var = Variable(torch.from_numpy(x_arr).float())
    return x_var[1:]

#train and test sets, normalized by mu and std of train sets
X_train_tensors_nao = slidingScaleX(nao[:testingBegin], xWindow, yWindow, leadDays)
mu_nao, std_nao = torch.mean(X_train_tensors_nao, axis=0), torch.std(X_train_tensors_nao, axis=0)
X_train_tensors_nao = (X_train_tensors_nao - mu_nao)/std_nao
X_test_tensors_nao = slidingScaleX(nao[testingBegin:], xWindow, yWindow, leadDays)
X_test_tensors_nao = (X_test_tensors_nao - mu_nao)/std_nao

X_train_tensors_m1 = slidingScaleX(m1[:testingBegin], xWindow, yWindow, leadDays)
mu_m1, std_m1 = torch.mean(X_train_tensors_m1, axis=0), torch.std(X_train_tensors_m1, axis=0)
X_train_tensors_m1 = (X_train_tensors_m1 - mu_m1)/std_m1
X_test_tensors_m1 = slidingScaleX(m1[testingBegin:], xWindow, yWindow, leadDays)
X_test_tensors_m1 = (X_test_tensors_m1 - mu_m1)/std_m1


X_train_tensors_m2 = slidingScaleX(m2[:testingBegin], xWindow, yWindow, leadDays)
mu_m2, std_m2 = torch.mean(X_train_tensors_m2, axis=0), torch.std(X_train_tensors_m2, axis=0)
X_train_tensors_m2 = (X_train_tensors_m2 - mu_m2)/std_m2
X_test_tensors_m2 = slidingScaleX(m2[testingBegin:], xWindow, yWindow, leadDays)
X_test_tensors_m2 = (X_test_tensors_m2 - mu_m2)/std_m2

X_train_tensors = torch.stack((X_train_tensors_nao, X_train_tensors_m1, X_train_tensors_m2), axis = 2)
X_test_tensors = torch.stack((X_test_tensors_nao, X_test_tensors_m1, X_test_tensors_m2), axis = 2)

#number of features used in input
numFeatures = X_train_tensors.shape[2]

#Next we must reshape our data to be suitable input for an LSTM
#Recall, LSTM's have input and output for EACH timestep
# So, the LSTM input should be in the form "(#samples, timestep, #features)"
#see: https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
X_train_tensors_final = X_train_tensors.reshape(X_train_tensors.shape[0], 1, -1)
X_test_tensors_final = X_test_tensors.reshape(X_test_tensors.shape[0], 1, -1)

#so we've created a new var with same data and number of elems as x_train tensors
#but now in the shape of a (testBegin - xWindow)*1*numFeatures*xWindow tensor


"""
Now we build the LSTM. Every model in PyTorch needs to be inherited from 
the nn.Module superclass. The LSTM consists of:

- 2 LSTM layers with the same hyperparameters stacked over each other 
- (via hidden_size),
- 2 Fully Connected layers, 
- the ReLU layer,
- and some helper variables

To get more detail on the nn.LSTM function, go here: 
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html


"""

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of output classes 
                                        #(i.e. output categories)

        self.num_layers = num_layers #number of stacked LSTM layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #num features in hidden state 
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) 
                          #lstm
                          #batch_first = True, so input/output tensors provided

        #Explanation of nn.Lienar and nn.ReLU given here:
        #https://ashwinhprasad.medium.com/pytorch-for-deep-learning-nn-linear-and-nn-relu-explained-77f3e1007dbb
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            #nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            #nn.Sigmoid()

            )

    
    #Defining the forward pass of the LSTM

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #hidden state, initialized with 0's

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #internal state, initialized with 0's

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        #feed lstm with input, hidden, and internal state at time t.
        #Returns and stores new versions

        hn = hn.view(-1, self.hidden_size) 
        #reshaping the data for Dense layer next

        #apply activations, and pass them to the dense layers, and return the 
        #output.
        return self.model(hn)


input_size = numFeatures*xWindow #dimension of the input
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 

#instantiate  the class LSTM1 object

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])


#Define loss function and optimizer

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 


########################
#     TRAINING
########################

#loop for the number of epochs, do the forward pass, calculate the loss,
#improve the weights via the optimizer step

for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0

  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)

  loss.backward() #calculates the loss of the loss function

  optimizer.step() #improve from loss, i.e backprop

  #Next we print the loss for every 100 epochs. You'll see the loss decrease, 
  #so the model's doing well

  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


###########################
#         TESTING
###########################


#Runs the LSTM on all training data and provides predicted results in an array, 
#along with an array of what the actual data values are. First predicted month
#is January of 2001 (because it first tests on Jan 2000 to Dec 2000)

#Input: original dataframe, beginning index of testing data (int), xWindow (int)
#     , number of months after the input window we predict (int)

#Output: an array of predicted values, and an array of actual values


def get_results(x_test):
    predicted = []
    
    for x in x_test:
        #run the lstm on this small bit of input

        predict = lstm1(x.reshape(1, 1, -1))
        predicted.append(predict.item())

    return predicted

predictedData = get_results(X_test_tensors_final)

realData = y_test_tensors

###########################################################
#         PLOTTING (so far, only intended for yWindow = 1)
###########################################################

"""
testingDates = df_with_dates.index[testingBegin + xWindow:]
testingDatesArr = []
for date in testingDates:
  testingDatesArr+= [str(int(date[1])) + "-" + str(int(date[0]))]
"""

#create the figure. Adjust size if needed
fig = plt.figure(figsize=(10,6)) 

#Create axes object
ax = fig.add_subplot(111)

#Adjust how many x-axis ticks there are
ax.xaxis.set_major_locator(plt.MaxNLocator(5))

scaler1 = preprocessing.MinMaxScaler()
n1=scaler1.fit_transform(realData.reshape(-1, 1)).flatten()
scaler2 = preprocessing.MinMaxScaler()
n2=scaler2.fit_transform(np.array(predictedData).reshape(-1, 1)).flatten()
yhat = savgol_filter(predictedData, w, p)
scaler3 = preprocessing.MinMaxScaler()
n3=scaler3.fit_transform(np.array(yhat).reshape(-1, 1)).flatten()

plt.plot(n1, label='Actual Data') #actual plot
#plt.plot(n2, '--',label='Predicted Data') #predicted plot
plt.plot(n3,label='Smoothed Prediction') #predicted plot

plt.legend()
plt.show()

corr1, p1 = stats.pearsonr(np.array(n1).flatten(), np.array(n2).flatten())
corr2, p2 = stats.pearsonr(np.array(n1).flatten(), np.array(n3).flatten())
print(corr1, corr2)

#print("Correlation when lag is 1 day and xWindow is " +
                                      #str(xWindow) + ": " + str(corr[0,1]) + "\n")

