import numpy as np
import pandas as pd 
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from bokeh.plotting import figure, show

#load the dataset
stock_dataset = pd.read_csv('GoogleClosingPrices.csv')
print(stock_dataset.head())
#Check the datatypes
print(stock_dataset.dtypes)
#Data transformation
#Convert date to datetime from original object
stock_dataset['date'] = pd.to_datetime(stock_dataset['date'])
#Convert closing price to float64 from original object
stock_dataset['close'] = pd.to_numeric(stock_dataset['close'], downcast='float')
#Check if the datatypes have changed
print(stock_dataset['date'].dtype)
print(stock_dataset['close'].dtype)
#Sorting stock data on date column
stock_dataset = stock_dataset.sort_values(by='date')
print(stock_dataset)
#Plot #1:Plotting close prices for all the days we have in our file
google_close_prices = stock_dataset['close']
dataset_length = len(google_close_prices)
fig_closingprice = figure(x_axis_label='Day', y_axis_label='Closing price')
fig_closingprice.line(range(1,dataset_length-1), google_close_prices, color='blue', line_width=1)
show(fig_closingprice)
#reshaping the dataset
google_close_prices = google_close_prices.values.reshape(dataset_length, 1)
#normalize data
scaler = MinMaxScaler(feature_range=(0,1))
google_close_prices = scaler.fit_transform(google_close_prices)
#split data into train and test 80:20
train_size = int(dataset_length * 0.8)
test_size = len(google_close_prices) - train_size
train_google, test_google = google_close_prices[0:train_size, :], google_close_prices[train_size:dataset_length, :]
#Choosing random seed value
np.random.seed(7)

print('Split data into train and test: ', len(train_google), len(test_google))
#Construct a time-series from the dataset for a given time window (e.g. 30 days)
def create_timeseries_from_data(dataset, interval):
    XAxis=[]
    YAxis=[]
    for i in range(len(dataset)-interval-1):
        #30 day closing price cycle
        XAxis.append(dataset[i:(i+interval), 0])
        #Next value after the cycle
        YAxis.append(dataset[i+interval, 0])
    return np.array(XAxis), np.array(YAxis)

#Time series interval is 30 (can be changed to improve performance further)
time_series_interval = 30
#Training data time series
x_train, y_train = create_timeseries_from_data(train_google, time_series_interval)
#Test data time series
x_test, y_test = create_timeseries_from_data(test_google, time_series_interval)
print(x_train.shape[0])
print(x_train.shape[1])
#Convert to LSTM layer input size, i.e. samples (all close prices), steps (30) and features (1)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#build the model
#start with a Sequential layer
model = Sequential()
#Adding LSTM recurrant layer
model.add(LSTM( 4, input_shape=(time_series_interval, 1)))
#Adding output layer with 1 output (i.e. predicted closing price)
model.add(Dense(1))
#We'll measure loss on root mean square error
model.compile(loss='mse', optimizer='adam')
#fit the model, 200 epochs, 32 samples per batch
epochs = 200
model_history = model.fit(x_train, y_train, epochs=epochs, batch_size=32)
#Printing summary of the model created
print(model.summary())
#testing our model
predict_train = model.predict(x_train)
predict_test = model.predict(x_test)
#Since the above predicted values are in normzalized form, we need to denormalize them to get
#the actual closing price
#predicted values
predict_train = scaler.inverse_transform(predict_train)
predict_test = scaler.inverse_transform(predict_test)
#original values
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])
#Calculating the final root mean square errors for our predicted values
rms_train = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
rms_test = math.sqrt(mean_squared_error(y_test[0], predict_test[:, 0]))
print('Training data Root Mean Squared Error:' + str(rms_train))
print('Test data Root Mean Squared Error:' + str(rms_test))
#Training data
train_plot = np.empty_like(google_close_prices)
train_plot[:,:] = np.nan
train_plot[time_series_interval:len(predict_train) + time_series_interval, :] = predict_train
#Test data
test_plot = np.empty_like(google_close_prices)
test_plot[:,:] = np.nan
test_plot[len(predict_train) + (time_series_interval * 2) + 1:len(google_close_prices) - 1, :] = predict_test
##Plot #2
##plot predicted values vs actual values
#Bokeh version
fig_predict = figure(x_axis_label='Day', y_axis_label='Closing price')
train_plot = np.reshape(train_plot,train_plot.size)
test_plot = np.reshape(test_plot,test_plot.size)
google_close_prices = scaler.inverse_transform(google_close_prices)
google_close_prices = np.reshape(google_close_prices,dataset_length)
fig_predict.line(range(1,dataset_length), google_close_prices,legend='Actual closing prices',color='blue', line_width=1)
fig_predict.line(range(1,dataset_length), train_plot,legend='Predicted training closing price',color='red', line_width=1)
fig_predict.line(range(1,dataset_length), test_plot,legend='Predicted test closing price',color='green', line_width=1)
fig_predict.legend.location = "top_left"
show(fig_predict)
#Bokeh: plot error rate for each epoch as well
fig_loss = figure(x_axis_label='Epochs', y_axis_label='Loss')
fig_loss.line(range(1,epochs+1), model_history.history["loss"], color='blue', line_width=1)
show(fig_loss)

