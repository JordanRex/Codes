## tensorboard codes

import tensorflow as tf

## Embedding Layer ##
# to create the embedding layer for the high cardinal categorical features
models=[]
inputs=[]

for cat in catemb:
    vocab_size = enc.train_full[cat].nunique()+1
    inpt = tf.keras.layers.Input(shape=(1,),\
                                 name='input_' + '_'.join(cat.split(' ')))
    embed = tf.keras.layers.Embedding(vocab_size, 200, trainable=True,
                                      embeddings_initializer=tf.initializers.RandomNormal())(inpt)
    embed_reshaped = tf.keras.layers.Reshape(target_shape=(200,))(embed)
    models.append(embed_reshaped)
    inputs.append(inpt)
    
## define the keras model
num_input = tf.keras.layers.Input(shape=(len(num)),name='input_num')
inputs.append(num_input)
models.append(num_input)
merge_models= tf.keras.layers.concatenate(models)
pre_preds = tf.keras.layers.Dense(1000, activation = 'relu')(merge_models)
pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
pre_preds = tf.keras.layers.Dense(1000, activation = 'relu')(pre_preds)
pre_preds = tf.keras.layers.BatchNormalization()(pre_preds)
pred = tf.keras.layers.Dense(1, activation = 'sigmoid')(pre_preds)
model_full = tf.keras.models.Model(inputs = inputs, outputs =pred)
model_full.compile(loss = tf.keras.losses.binary_crossentropy,\
                   metrics = ['accuracy'],
                   optimizer = 'adam')

log_dir = './logs/fit' #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

input_dict = {
    'input_cat1': np.array(enc.train_enc[catemb[0]]),
    "input_cat2": np.array(enc.train_enc[catemb[1]]),
    "input_NAME": np.array(enc.train_enc[catemb[2]]),
    "input_num": np.array(enc.train_enc[num])
}

model_full.fit(input_dict,enc.ytrain_enc*1,epochs=10,batch_size=100,class_weight={0:0.25,1:0.75}, 
          callbacks=[tensorboard_callback])
print('tf model is fit', '\n')

test_dict = {
    'input_cat1': np.array(enc.test_enc[catemb[0]]),
    "input_cat2": np.array(enc.test_enc[catemb[1]]),
    "input_num": np.array(enc.test_enc[num])
}

testpred = model_full.predict(test_dict)
print('tf predictions are made', '\n')
pred_labels = np.where(testpred>0.45, 1, 0)

print(skm.roc_auc_score(y_score=testpred, y_true=enc.ytest_enc))
print(skm.confusion_matrix(y_pred=pred_labels, y_true=enc.ytest_enc))
print(skm.recall_score(y_pred=pred_labels, y_true=enc.ytest_enc))
print(skm.precision_score(y_pred=pred_labels, y_true=enc.ytest_enc))

# %tensorboard --logdir=='./logs/fit/train'
