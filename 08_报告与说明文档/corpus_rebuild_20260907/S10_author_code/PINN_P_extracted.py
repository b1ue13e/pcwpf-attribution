# Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import sys
sys.path.insert(0,'../')  # add previous directory to path to load constants module
import constants as const

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
print(tf.__version__)

import sciann as sn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
print(sn.__version__)


# Import dataset using pandas

raw_dataset = pd.read_csv('../../Dataset/Dataset_reduced.csv')
dataset = raw_dataset.copy()
print("Shape of dataset: "+str(dataset.shape))
dataset.tail()

# Clean dataset of NaN values
#dataset.isna().sum()
print(dataset.isna().sum())
dataset = dataset.dropna()

# One-hot encode categorical columns

dataset['turbine'] = dataset['turbine'].map({1:'R80711', 2:'R80721',3:'R80736', 4:'R80790'})
dataset = pd.get_dummies(dataset, columns=['turbine'], prefix='', prefix_sep='')
dataset.tail()

# Split the data into training and test sets

train_dataset = dataset.sample(frac=0.8, random_state=10)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data with the joint distribition of a few pairs of columns from the training set

sns.pairplot(dataset[['P', 'T', 'V', 'theta', 'W']], diag_kind='kde')


# Check overal statistis of each input feature 
dataset.describe().transpose()

# Normalization

print(dataset.describe().transpose()[['mean', 'std']])

train_dataset_norm=(train_dataset-dataset.mean())/dataset.std()
test_dataset_norm=(test_dataset-dataset.mean())/dataset.std()

print()
print("Normalized dataset")
print(train_dataset_norm.describe().transpose()[['mean', 'std']])

# Functions to normalize and compute original units from normalized units

T_mean = dataset['T'].mean()
T_std = dataset['T'].std()
W_mean = dataset['W'].mean()
W_std = dataset['W'].std()
P_mean = dataset['P'].mean()
P_std = dataset['P'].mean()

def normalize_P(P):
    return (P-P_mean)/P_std
def normalize_T(T):
    return (T-T_mean)/T_std
def normalize_W(W):
    return (W-W_mean)/W_std

def units_P(P):
    return P*P_std + P_mean
def units_T(T):
    return T*T_std + T_mean
def units_W(W):
    return W*W_std + W_mean

# Split features from labels/targets

""" train_dataset_copy = train_dataset_norm.copy()
test_dataset_copy = test_dataset_norm.copy()

train_features = train_dataset_copy[['V','theta','W','P']]
test_features = test_dataset_copy[['V','theta','W','P']]

train_labels = train_dataset_copy['T']
test_labels = train_dataset_copy['T']

print(train_features.shape)
print(train_labels.shape) """

# Function to compute torque from P using the physical equation P = g * T * W

def P_fromT(W,T):
    Pturbina = const.gear * W * T
    return Pturbina

# Set up the NN arquitecture to fit the data

# Independent variables
Vw = sn.Variable('Vw', dtype='float64')
theta = sn.Variable('theta', dtype='float64')
Wt = sn.Variable('Wt', dtype='float64')
Torque = sn.Variable('Torque', dtype='float64')

# Solution variables
Power = sn.Functional('Power',variables=[Vw,theta,Wt],hidden_layers=[64,64],activation='relu')

# Define targets (data and constrains)

d1 = sn.Data(Power)
c1 = sn.Tie( Power, normalize_P(P_fromT(units_W(Wt),units_T(Torque))) )
# Define optimization model
model = sn.SciModel(
    #inputs=[Vw,theta,Wt,Power],
    inputs=[Vw,theta,Wt,Torque],
    targets=[d1,c1],
    loss_func='mae',
    #loss_func='mse',
    optimizer='adam',
    plot_to_file='../../Models/PINN_Power.png'
)

model.summary()



# Train the model

X_train = train_dataset_norm[['V','theta','W','T']].to_numpy()
input_data = [X_train[:,0],X_train[:,1],X_train[:,2],X_train[:,3]]

Y_train = train_dataset_norm['P'].to_numpy()
data_d1 = Y_train[:]
data_c1 = 'zeros'
target_data = [data_d1,data_c1]

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=10, min_lr=1e-8)

running = model.train(x_true=input_data,y_true=target_data,
                      batch_size=128,learning_rate=0.001, epochs=200,verbose=1, callbacks=[reduce_lr])

plt.xlabel('Epochs',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.plot(running.history['loss'],label='loss')
plt.show()
plt.plot(running.history['Power_loss'],label='Power_loss')
plt.plot(running.history['sub_2_loss'],label='Physical_loss')
plt.legend()
plt.show()

# Evaluation of the model on Power data

# Pred vs True values

X_test = test_dataset_norm[['V','theta','W','T']].to_numpy()
Y_test = test_dataset_norm['P'].to_numpy()

y_true = Y_test[:]
y_pred = Power.eval([X_test[:,0],X_test[:,1],X_test[:,2]])

a = plt.axes(aspect='equal')
p1 = max(max(y_true), max(y_true))
p2 = min(min(y_true), min(y_true))
plt.plot([p1, p2], [p1, p2], 'r--')
plt.scatter(y_true, y_pred)
plt.xlabel('True Values [P]')
plt.ylabel('Predictions [P]')
plt.show()
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)

print("Test MAE:", mean_absolute_error(y_true,y_pred))


# Evaluate P on original units

# Pred vs True values

X_test = test_dataset_norm[['V','theta','W','T']].to_numpy()
Y_test = test_dataset_norm['P'].to_numpy()
y_true = Y_test[:]
y_pred = Power.eval([X_test[:,0],X_test[:,1],X_test[:,2]])

x_mean = np.nanmean( dataset, axis=0)
x_std = np.nanstd( dataset, axis=0 )
column_name = dataset.columns.values
print(column_name)
Power_column = 4 

y_pred = y_pred * x_std[4] + x_mean[4]
y_true = y_true * x_std[4] + x_mean[4]

a = plt.axes(aspect='equal')
p1 = max(max(y_pred), max(y_true))
p2 = min(min(y_pred), min(y_true))
plt.plot([p1, p2], [p1, p2], 'r--')
plt.scatter(y_true, y_pred)
plt.xlabel('True Values [P]')
plt.ylabel('Predictions [P]')
plt.show()
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)

p_cutoff = 25000
y_pred_nonzero = y_pred[abs(y_true)>p_cutoff]
y_true_nonzero = y_true[abs(y_true)>p_cutoff]

test_mae = mean_absolute_error(y_true,y_pred)
test_rmse = mean_squared_error(y_true,y_pred,squared=False)
#test_mape = mean_absolute_percentage_error(y_true,y_pred)*100
test_mape = mean_absolute_percentage_error(y_true_nonzero,y_pred_nonzero)*100
test_r2 = r2_score(y_true,y_pred)

print()
print("Test MAE: "+str(test_mae))
print("Test RMSE: "+str(test_rmse))
print("Test MAPE: "+str(test_mape)+" %")
print("Test R2: "+str(test_r2))

# save model weights 
model.save_weights('../../Models/PINN_P_weights.hdf5')

# load model weights
#model.load_weights('../../Models/Torque_model_weights.hdf5')

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
#model.save('../../Models/PINN_P_model.h5')

# Recreate the exact same model, including its weights and the optimizer
#new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
#new_model.summary()

#load model weights
model.load_weights('../../Models/PINN_P_weights.hdf5')

#model = tf.keras.models.load_model('../../Models/Power_model.h5')
model.summary()

# Evaluate P on P_fromT (g*T*W)

# Pred vs True values

X_test = test_dataset_norm[['V','theta','W','T']].to_numpy()
Y_test = test_dataset_norm['P'].to_numpy()
#y_true = Y_test[:]
W_true = test_dataset['W'].to_numpy()
T_true = test_dataset['T'].to_numpy()
y_true = const.gear * T_true * W_true
#print(y_true.shape)
y_pred = Power.eval([X_test[:,0],X_test[:,1],X_test[:,2]])

x_mean = np.nanmean( dataset, axis=0)
x_std = np.nanstd( dataset, axis=0 )
column_name = dataset.columns.values
print(column_name)
Power_column = 4 

y_pred = y_pred * x_std[4] + x_mean[4]
#y_true = y_true * x_std[4] + x_mean[4]

a = plt.axes(aspect='equal')
p1 = max(max(y_pred), max(y_true))
p2 = min(min(y_pred), min(y_true))
plt.plot([p1, p2], [p1, p2], 'r--')
plt.scatter(y_true, y_pred)
plt.xlabel('True Values [P]')
plt.ylabel('Predictions [P]')
plt.show()
#lims = [0, 50]
#plt.xlim(lims)
#plt.ylim(lims)
#_ = plt.plot(lims, lims)

p_cutoff = 25000
y_pred_nonzero = y_pred[abs(y_true)>p_cutoff]
y_true_nonzero = y_true[abs(y_true)>p_cutoff]

test_mae = mean_absolute_error(y_true,y_pred)
test_rmse = mean_squared_error(y_true,y_pred,squared=False)
#test_mape = mean_absolute_percentage_error(y_true,y_pred)*100
test_mape = mean_absolute_percentage_error(y_true_nonzero,y_pred_nonzero)*100
test_r2 = r2_score(y_true,y_pred)

print()
print("Test MAE: "+str(test_mae))
print("Test RMSE: "+str(test_rmse))
print("Test MAPE: "+str(test_mape)+" %")
print("Test R2: "+str(test_r2))