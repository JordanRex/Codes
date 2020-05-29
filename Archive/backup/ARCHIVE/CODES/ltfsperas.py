## ltfs_peras.py

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

def data():
    import pickle
    
    # load backup
    car = open('./ltfs.pkl', 'rb')
    X = pickle.load(car)
    Y = pickle.load(car)
    XV = pickle.load(car)
    YV = pickle.load(car)
    car.close()
    return X, Y, XV, YV

def create_model(X, Y, XV, YV):
    # clear session each time to ensure it is a fresh run
    tf.keras.backend.clear_session()
    
    input_dim = X.shape[1]

    model = Sequential()
    model.add(Dense(input_dim, input_dim = input_dim , activation={{choice(['relu', 'tanh'])}}))
    model.add(BatchNormalization())
    model.add(Dense({{choice([2000])}}, activation={{choice(['relu'])}}))
    model.add(Dropout({{uniform(0, 0.2)}}))
    model.add(Dense({{choice([250, 500])}}, activation = {{choice(['relu'])}}))
    if {{choice(['true', 'false'])}} == 'true':
        model.add(Dense({{choice([5, 10, 25])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(Dropout({{uniform(0, 0.2)}}))
    model.add(Dense(1, activation={{choice(['sigmoid'])}}))

    model.compile(loss='binary_crossentropy', optimizer = {{choice([tf.keras.optimizers.Adam(learning_rate=0.01)])}}, 
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    model.fit(X, Y, batch_size={{choice([100])}}, epochs=5, verbose=2, validation_data=(XV, YV), shuffle=True, 
              callbacks=[EarlyStopping(monitor='val_auc', patience=2, verbose=0, mode='max')])
    score, acc, auc = model.evaluate(XV, YV, verbose=0)
    print('Test auc:', auc)
    return {'loss': 1-auc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=create_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=2,
                                                 trials=Trials(),
                                                 eval_space=True,
                                                 return_space=True)
    X, Y, XV, YV = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(XV, YV))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('model.h5')
