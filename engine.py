from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import datetime
from pytz import timezone
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
#import MySQLdb

france = timezone('Europe/France')


#from twilio.rest import Client

# Your Account SID from twilio.com/console
#account_sid = "*******"
# Your Auth Token from twilio.com/console
#auth_token  = "*******"

#client = Client(account_sid, auth_token)

# Merge Datas
def merge_data(bitcoin_file, tweet_file):
	df_btc = pd.read_csv(bitcoin_file)
	df_btc.columns = ["Price","24h_volume_usd","market_cap_usd","available_supply","total_supply","percent_change_1h","percent_change_24h","percent_change_7d", "sell", "buy", "15m", "Time"]
	df_tweet = pd.read_csv(tweet_file)
	df_tweet.columns = ["Sentiment", "Time"]

	df_merged_data = df_btc.merge(df_tweet, how='inner', on='Time')
	df_merged_data.to_csv("merged_data.csv", index=None)

	return "merged_data.csv"


def create_dataset(dataset, look_back, sentiment, sent=False):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        if i >= look_back:
            a = dataset[i-look_back:i+1, 0]
            a = a.tolist()
            if(sent==True):
                a.append(sentiment[i].tolist()[0])
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
    #print(len(dataY))
    return np.array(dataX), np.array(dataY)


def train_test_split(merged_file):
	data = pd.read_csv(merged_file)
	datag = data[['Price','Sentiment']].groupby(data['Time']).mean()

	# Prepare data
	values = datag['Price'].values.reshape(-1,1)
	sentiment = datag['Sentiment'].values.reshape(-1,1)
	values = values.astype('float32')
	sentiment = sentiment.astype('float32')
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)

	train_size = int(len(scaled) * 0.7)
	test_size = len(scaled) - train_size
	train_price, test_price = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
	train_sent, test_sent = sentiment[0:train_size], sentiment[train_size:len(scaled)]
	print(len(train_price), len(test_price))
	split = train_size

	look_back = 2
	trainX, trainY = create_dataset(train_price, look_back, train_sent,sent=True)
	testX, testY = create_dataset(test_price, look_back, test_sent, sent=True)

	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	return trainX, trainY, testX, testY, scaler

def train_LSTM(trainX, trainY, testX, testY, scaler):
	model = Sequential()
	model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
	model.add(LSTM(100))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

	yhat = model.predict(testX)

	yhat_inverse_sent = scaler.inverse_transform(yhat.reshape(-1, 1))
	testY_inverse_sent = scaler.inverse_transform(testY.reshape(-1, 1))

	rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, yhat_inverse_sent))
	print('Test RMSE: %.3f' % rmse_sent)

	return model

def process_data(in_data):
    out_data = []
    for line in in_data:
        out_data.append(float(line.split(',')[0]))
    return np.array(out_data).reshape(-1,1)

def predict(model, scaler):
	#Enter the values for you database connection
	dsn_database = "bitcoin"         # e.g. "MySQLdbtest"
	dsn_hostname = "127.0.0.1"      # e.g.: "mydbinstance.xyz.us-east-1.rds.amazonaws.com"
	dsn_port = 3306                  # e.g. 3306 
	dsn_uid = "vico"             # e.g. "user1"
	dsn_pwd = "Sn2Sdum45!"              # e.g. "Password123"

	conn = mysql.connector.connect(host=dsn_hostname, port=dsn_port, user=dsn_uid, passwd=dsn_pwd, db=dsn_database)
	#conn = MySQLdb.connect(host=dsn_hostname, port=dsn_port, user=dsn_uid, passwd=dsn_pwd, db=dsn_database)

	cursor=conn.cursor()


	import queue 
	import time

	import queue
	import matplotlib.pyplot as plt
	true_q = queue.Queue()
	pred_q = queue.Queue()
	'''
	fig = plt.figure()
	ax = fig.add_subplot(111)
	fig.show()
	fig.canvas.draw()
	plt.ion()
	'''

	prev = 15000
	threshold = 0.05
	while True:
	    btc = open('live_bitcoin.csv','r')
	    sent = open('live_tweet.csv','r')
	    bit_data = btc.readlines()
	    sent_data = sent.readlines()
	    bit_data = process_data(bit_data[len(bit_data)-5:])
	    sent_data = process_data(sent_data[len(sent_data)-5:])
	    live = scaler.transform(bit_data)
	    testX, testY = create_dataset(live, 2, sent_data, sent=True)
	    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	    yhat = model.predict(testX)
	    yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
	    true_q.put(bit_data[4])
	    pred_q.put(yhat_inverse[0])
	    val = 100*((yhat_inverse[0][0] - prev)/prev)
	    if val > threshold:
	        decision = 'Buy!!!'
	        #message = client.messages.create(to="+15184234418‬", from_="+15188883052", body=decision+' - Price of Bitcoin is expected to rise.')
	    elif val <-threshold:
	        decision = 'Sell!!!'
	        #message = client.messages.create(to="+15184234418", from_="+15188883052", body=decision+' - Price of Bitcoin is expected to drop.')
	    else:
	        decision = ''
	    print(decision)
	    prev = yhat_inverse[0][0]
	    input_string = "INSERT INTO live_data values ({},{},{},'{}','{}');".format(yhat_inverse[0][0],bit_data[0][0],sent_data[4][0],datetime.datetime.now(tz=france).strftime('%Y-%m-%d %H:%M:%S'),decision)
	    print(input_string)
	    cursor.execute(input_string)
	    conn.commit()
	    time.sleep(60)
    

if __name__ == '__main__':
	merge_file = merge_data('live_bitcoin.csv', 'live_tweet.csv')
	trainX, trainY, testX, testY, scaler = train_test_split(merge_file)
	model = train_LSTM(trainX, trainY, testX, testY, scaler)
	predict(model, scaler)
