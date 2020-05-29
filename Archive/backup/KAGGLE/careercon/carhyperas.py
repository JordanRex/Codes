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
    car = open('./car.pkl', 'rb')
    X = pickle.load(car)
    Y = pickle.load(car)
    XV = pickle.load(car)
    YV = pickle.load(car)
    car.close()
    return X, Y, XV, YV

def create_model(X, Y, XV, YV):
    input_dim = X.shape[1]

    model = Sequential()
    model.add(Dense(input_dim, input_dim = input_dim , activation={{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(BatchNormalization())
    model.add(Dense({{choice([50, 100, 250, 500, 1000, 2000])}}, activation={{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(Dropout({{uniform(0, 0.7)}}))
    model.add(Dense({{choice([50, 100, 250, 500, 1000])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    if {{choice(['true', 'false'])}} == 'true':
        model.add(Dense({{choice([5, 20, 30, 50, 100])}}, activation = {{choice(['relu', 'sigmoid', 'tanh', 'elu'])}}))
    model.add(Dropout({{uniform(0, 0.7)}}))
    model.add(Dense(9, activation={{choice(['softmax'])}}))

    model.compile(loss='categorical_crossentropy', optimizer = {{choice(['rmsprop', 'adam', 'sgd', 'nadam', 'adadelta'])}}, 
                  metrics=['accuracy'])
    model.fit(X, Y, batch_size={{choice([20, 50])}}, epochs=10, verbose=2, validation_data=(XV, YV), shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.001)])
    score, acc = model.evaluate(XV, YV, verbose=1)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=create_model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=10,
                                                 trials=Trials(),
                                                 eval_space=True,
                                                 return_space=True)
    X, Y, XV, YV = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(XV, YV))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('model.h5')
    