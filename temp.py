import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.compose
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.optimizers import Adam
import numpy as np

dataset = pd.read_csv('life_expectancy.csv')
dataset = dataset.drop(['Country'],axis = 1)
features = dataset.iloc[:,0:-1]
labels = dataset.iloc[:,-1]

features = pd.get_dummies(features) #one-hot encoding

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=25)


#extracting numerical data from dataset
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = sklearn.compose.ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

features_train_scaled = np.asarray(features_train_scaled).astype(np.float32)
labels_train = np.asarray(labels_train).astype(np.float32)    
features_test_scaled = np.asarray(features_test_scaled).astype(np.float32)    
labels_test = np.asarray(labels_test).astype(np.float32)

my_model = Sequential()
my_model.add(
  InputLayer(
    input_shape = (features_train_scaled.shape[1],)
  )
)
my_model.add(Dense(3,activation="relu"))
my_model.add(Dense(1))
my_model.summary()
opt = Adam(learning_rate = 0.01)
my_model.compile(loss='mse',metrics=['mae'],optimizer=opt)

#train
my_model.fit(features_train_scaled,labels_train,epochs=40,batch_size=1,verbose=1)
# my_model.fit(data, labels, epochs = 20, batch_size = 1, verbose = 1,  validation_split = 0.2)
res_mse,res_mae = my_model.evaluate(features_test_scaled,label_test,verbose=0)

print(res_mse)
print(res_mae)



