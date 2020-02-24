import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tensorflow
import pandas as pandas
from tensorflow import keras
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense


try:
  user_data = pandas.read_csv("users.csv", delimiter = ";")
  print("user data loaded succesfully")
except:
  print("error loading user data")


# data with x values and target user count columns
#all columns needed (how x affects L1547777 user count?)
x_parameters_usercount_target_colums = [x for x in user_data 
if x.endswith('_x') | x.endswith('L1547777')]

x_parameters_columns= [x for x in user_data 
if x.endswith('our') | x.endswith('_x')]

x_parameters = user_data[x_parameters_columns]

# inputs
inputs = np.array(user_data[x_parameters_columns])
print("inputs", inputs)



# user count cell for target cell
user_counts_target_column = [x for x in user_data if x.endswith('L1547777')]


# matrix for x variables and avg user count of target
# here all needed values
x_values = user_data[x_parameters_usercount_target_colums]
print("whole matrix" , x_values)


# type of target L1547777 is object. Need to change it to float 
x_values.L1547777= x_values.L1547777.str.replace(',','.')
x_values.L1547777 = x_values.L1547777.astype(float)


#outputs
user_data['L1547777']=user_data['L1547777'].str.replace(',','.')
outputs = user_data[user_counts_target_column].astype(float)
outputs = np.array(outputs)
print("outputs: ",outputs)


epochs = 300 # number of rounds (forward and backward)
batch_size = 32 # samples to put at once
def define_model():
  model = Sequential()
  model.add(Dense(64, input_shape=(10,), activation='relu')) #hidden layer and inputs size 10
  model.add(Dense(1)) #output layer, output size 1
  print(model.summary())
  model.compile(loss='mse', optimizer='adam')
  return model

def train_model(inp, outp, model, epochs, batch_size):
  h = model.fit(inp, outp, validation_split=0.2,
  epochs=epochs,
  batch_size=batch_size,
  verbose=1)
  plt.figure(figsize=(15,2.6))
  plt.plot(h.history['loss'])
  plt.title(u"Loss during training", fontweight='bold', fontsize=20)
  plt.ylabel('Loss')
  plt.xlabel('Training rounds')

  return model

model = define_model()
model = train_model(inputs, outputs, model, epochs, batch_size)

plt.show()

def predict_model():
  pred = model.predict(inputs, batch_size=500)
  
  plt.title(u"Distribution of avg user counts", fontweight='bold', fontsize=20)
  sns.distplot(outputs)
  sns.distplot(pred)
  plt.legend(['Original user counts', 'Predictions'])
  plt.show()

  plt.figure(figsize=(11,5))
  plt.ylabel('Average user count')
  plt.xlabel('Measurement time')
  plt.plot(pred, color="blue", marker='.')
  plt.plot(outputs, color="green", marker='.')
  plt.title(u"Actual outputs and predictions", fontweight='bold', fontsize=20)
  plt.legend(['Predictions', 'Original user counts'])

  plt.show()

predict_model()


