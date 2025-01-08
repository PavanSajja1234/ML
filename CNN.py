import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Conv2DTranspose, BatchNormalization, Activation, Conv2D, Flatten, \
    Conv1D, GlobalMaxPooling1D, Conv3D, GlobalMaxPooling3D, LSTM, MaxPooling2D
from tensorflow.keras.utils import to_categorical
#from keras_preprocessing import sequence
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# this is deep learning using activation function as softmax on dataset.

dataset_path = 'dia_betes.csv'

# load the dataset
dataset = np.loadtxt(dataset_path, delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# split training_set = 80%, test_set = 20%
# use random_state for shuffle dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=1)

# train_set = 60%, validation_set = 20% , test_set = 20%
# create validation_set by split 20% of training_set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=2)

# create zero column with numpy
X_train = np.column_stack((X_train, np.zeros((len(X_train),1))))
X_test = np.column_stack((X_test, np.zeros((len(X_test),1))))
X_val = np.column_stack((X_val, np.zeros((len(X_val),1))))

# reshape input data to be able do extract feature.
# reshape for CNN (instance, row, column, channel)
X_train = X_train.reshape(len(X_train),3,3,1)
X_test = X_test.reshape(len(X_test),3,3,1)
X_val = X_val.reshape(len(X_val),3,3,1)

# convert output class to category
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# define the keras model
model = Sequential()

model.add(Conv2D(64, 2, activation='relu', input_shape=(3,3,1)))
model.add(MaxPooling2D(pool_size=1))
model.add(Conv2D(32, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
# summarize of model (explanation model layer)
print(model.summary())

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, validation_data=(X_val, y_val,), epochs=650, batch_size=32, verbose=2)

# evaluate the model
scores = model.evaluate(X_train, y_train, verbose=0)
print('====================')
print('Result of validation set')
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print('====================')

# predict using test_set
value = model.predict(X_test)
y_pred = np.argmax(value,axis=1)
y_true = np.argmax(y_test,axis=1)

# confusion matrix
result_CF_matrix = confusion_matrix(y_true, y_pred, labels=[0,1])
TP = result_CF_matrix[0,0]
FN = result_CF_matrix[0,1]
FP = result_CF_matrix[1,0]
TN = result_CF_matrix[1,1]
print('              Confusion Matrix')
print('                Yes      No')
print('Actual Yes  %6d' %TP+'   %6d'%FN)
print('Actual No   %6d' %FP+'   %6d'%TN)

# Evaluation method from confusion matrix
ACC = (TP + TN)/(TP+FN+FP+TN)
Sensitivity = TP/(TP+FP)
Specificity = TN/(FN+TN)
Recall = TP/(TP+FN)

print('====================')
print('Result of testing set')
print('Accuracy = %.2f%%' % (ACC*100))
print('Sensiticity = %.2f%%' % (Sensitivity*100))
print('Specificity = %.2f%%' % (Specificity*100))
print('Precision = %.2f%%' % (Sensitivity*100))
print('Recall = %.2f%%' % (Recall*100))
print('====================')
