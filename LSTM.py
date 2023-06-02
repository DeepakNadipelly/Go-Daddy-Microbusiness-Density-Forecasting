import time
import requests
import json
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta 
import time
import copy
import random
import numpy as np
import os
from collections import defaultdict
import math
global logger
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
# from tqdm.notebook import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 1999
seed_everything(seed)

class CFG:
    quick_experiment = False
    epochs = 50 if quick_experiment else 200
    max_trials = 1 if quick_experiment else 30
    is_tuning = False
    is_training = False
    is_inference = True
    tuning_epochs = 10
    batch_size = 1024
    sequence_length = 8
    validation_length = 4
    target_field = "microbusiness_density"

def smape(y_true, y_pred):
    denominator = (y_true + tf.abs(y_pred)) / 200.0
    diff = tf.abs(y_true - y_pred) / denominator
    diff = tf.where(denominator == 0, 0.0, diff)
    return tf.reduce_mean(diff)


def get_cosine_decay_learning_rate_scheduler(epochs, lr_start=0.001, lr_end=1e-6):
    def cosine_decay(epoch):
        if epoch <= CFG.tuning_epochs:
            return lr_start
        if epochs > 1:
            w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
        else:
            w = 1
        return w * lr_start + (1 - w) * lr_end
    return LearningRateScheduler(cosine_decay, verbose=0)

def cauclate_smape(item):
    cfips = item.iloc[0].cfips
    y_true = tf.constant(item["microbusiness_density"], dtype=tf.float64)
    y_pred = tf.constant(item["prediction"], dtype=tf.float64)
    return smape(y_true, y_pred).numpy()
def diff_month(dt):
    return (dt.year - 2019) * 12 + dt.month - 8

start_time_here = time.time()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
revealed_test = pd.read_csv("revealed_test.csv")
revealed_test_dates = revealed_test.first_day_of_month.unique()
train = pd.concat([train, revealed_test])
train.index = np.arange(0, len(train))
census = pd.read_csv("census_starter.csv")
zero_cfips = list(train[train.microbusiness_density == 0].cfips.unique())
for cfips in zero_cfips:
    train.loc[train.cfips==cfips, CFG.target_field] = train[train.cfips==cfips].iloc[-1][CFG.target_field]

county_lookup_dict = dict()
unique_cfips = train.cfips.unique()
for i in range(len(unique_cfips)):
    county_lookup_dict[unique_cfips[i]] = i 
state_lookup_dict = dict()
unique_states = train.state.unique()
for i in range(len(unique_states)):
    state_lookup_dict[unique_states[i]] = i
train['x'] = pd.to_datetime(train['first_day_of_month']).apply(diff_month)
test['x'] = pd.to_datetime(test['first_day_of_month']).apply(diff_month)
train['county_id'] = train['cfips'].apply(lambda cfips: county_lookup_dict[cfips])
test['county_id'] =  test['cfips'].apply(lambda cfips: county_lookup_dict[cfips])
train['state_id'] = train['state'].apply(lambda state: state_lookup_dict[state])
last_value_dict = dict()
county_state_dict = dict()
for i in range(len(census)):
    item = census.iloc[i]
    cfips = item.cfips
    df = train[train.cfips == cfips]
    if len(df) == 0:
        continue
    y_values = list(df[CFG.target_field])
    county_state_dict[item.cfips] = df.iloc[0].state
    last_value = y_values[-1]
    last_value_dict[item.cfips] = last_value
test['state_id'] = test['cfips'].apply(lambda cfips: state_lookup_dict[county_state_dict[cfips]])

def preprocess(window):
    return window[:-1, :], window[-1, -1]
def make_dataset(df, sequence_length=CFG.sequence_length):
    dataset = tf.data.Dataset.from_tensor_slices((df[all_columns]))
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))
    dataset = dataset.map(preprocess)
    return dataset

embedding_columns = ["county_id", "state_id"]
all_columns = embedding_columns + [CFG.target_field]
samples_per_item = len(train.first_day_of_month.unique())
train_length = samples_per_item - CFG.sequence_length - CFG.validation_length
train_features = []
train_targets = []
valid_features = []
valid_targets = []
unique_cfips = list(train.cfips.unique())
last_targets = []
for cfips in unique_cfips:
    ds = make_dataset(train[train.cfips == cfips])
    for X, y in ds.batch(samples_per_item).take(1):
        train_features.append(X[:train_length])
        valid_features.append(X[train_length:])
        train_targets.append(y[:train_length])
        valid_targets.append(y[train_length:])
        last_targets.append(y[-CFG.sequence_length:])
train_feature_tensor = tf.concat(train_features, axis=0)
valid_feature_tensor = tf.concat(valid_features, axis=0)
train_target_tensor = tf.concat(train_targets, axis=0)
valid_target_tensor = tf.concat(valid_targets, axis=0)
train_feature_tensor.shape, valid_feature_tensor.shape, train_target_tensor.shape, valid_target_tensor.shape

train_ds = tf.data.Dataset.from_tensor_slices((train_feature_tensor, train_target_tensor))
train_ds = train_ds.shuffle(CFG.batch_size * 4).batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)
valid_ds = tf.data.Dataset.from_tensor_slices((valid_feature_tensor, valid_target_tensor))
valid_ds = valid_ds.batch(CFG.batch_size).cache().prefetch(tf.data.AUTOTUNE)

for X, y in train_ds.take(1):
    print(X.shape, y.shape)


def build_model_with_params(params):
    activation = "relu"
    use_dropout = params["use_dropout"]
    use_county_embedding = params["use_county_embedding"]
    use_state_embedding = params["use_state_embedding"]
    dropout_value = params["dropout"]
    learning_rate = params["learning_rate"]
    l2_factor = params["l2"]
    county_embed_size = params["county_embed_size"]
    state_embed_size = params["state_embed_size"]
    units = params["units"]
    inputs = tf.keras.Input(shape=(CFG.sequence_length, 3), dtype=tf.float32)
    county_inputs = inputs[:, 0, 0]
    state_inputs = inputs[:, 0, 1]
    target_inputs =inputs[:, :, 2]
    
    
    target_vector = tf.keras.layers.Reshape((CFG.sequence_length, 1))(target_inputs)
    
    sequence_vectors = []
    if use_county_embedding:
        county_vector = tf.keras.layers.Reshape((-1, 1))(county_inputs)
        county_vector = tf.keras.layers.Embedding(len(county_lookup_dict), county_embed_size, input_length=1)(county_vector)
        county_vector = tf.keras.layers.Reshape((county_embed_size, 1))(county_vector)
        sequence_vectors.append(county_vector)
    if use_state_embedding:
        state_vector = tf.keras.layers.Reshape((-1, 1))(state_inputs)
        state_vector = tf.keras.layers.Embedding(len(state_lookup_dict), state_embed_size, input_length=1)(state_vector)
        state_vector = tf.keras.layers.Reshape((state_embed_size, 1))(state_vector)
        sequence_vectors.append(state_vector)
    if len(sequence_vectors) > 0:
        sequence_vectors.append(target_vector)
        sequence_vector = tf.keras.layers.Concatenate(axis=-2)(sequence_vectors)
    else:
        sequence_vector = target_vector

    for i in range(len(units)):
        sequence_vector = tf.keras.layers.LSTM(
            units[i], 
            return_sequences=i != len(units) - 1, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_factor)
        )(sequence_vector)
    if use_dropout:
        sequence_vector = tf.keras.layers.Dropout(dropout_value)(sequence_vector)
    output = tf.keras.layers.Dense(1)(sequence_vector)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(loss="huber_loss", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=[smape])
    return model

def build_model_with_keras_tuner(hp):
    return build_model_with_params({
            "use_dropout": hp.Choice("use_dropout", [True, False]),
            "use_county_embedding": hp.Choice("use_county_embedding", [True, False]),
            "use_state_embedding": hp.Choice("use_state_embedding", [True, False]),
            "dropout": hp.Float("dropout", min_value=0.1, max_value=0.3),
            "learning_rate": hp.Float("learing_rate", min_value=1e-5, max_value=1e-3, sampling="log"),
            "l2": hp.Choice("l2", [1e-5, 3e-5, 5e-5, 1e-6, 5e-6]),
            "county_embed_size": hp.Choice("county_embed_size", [32, 64]),
            "state_embed_size": hp.Choice("state_embed_size", [8, 16]),
            "units": list(reversed(sorted([hp.Int(f"unit_{i}", min_value=16, max_value=128, step=16) for i in range(4)])))
    })
    
    
def build_model_with_best_params():
    return build_model_with_params({
            "use_dropout": False,
            "dropout": 0,
            "learning_rate": 1e-3,
            "l2": 5e-06,
            "county_embed_size": 32,
            "state_embed_size": 8,
            "units": [128, 128, 128, 128],
            "use_county_embedding": True,
            "use_state_embedding": False
    })

if CFG.is_tuning:
    tuner = kt.BayesianOptimization(
        build_model_with_keras_tuner,
        objective=kt.Objective("val_smape", direction="min"),
        max_trials=CFG.max_trials,
        overwrite=True
    )
    tuner.search(
        train_ds, 
        epochs=10, 
        validation_data=valid_ds, 
        verbose=2
    )


if CFG.is_tuning:
    tuner.results_summary()

custom_objects={
    "smape": smape
}
models = []
if CFG.is_tuning:
    model = tuner.get_best_models()[0]
    models.append(model)
models.append(tf.keras.models.load_model("model.tf", custom_objects=custom_objects))
models.append(tf.keras.models.load_model("best_model.tf", custom_objects=custom_objects))

if CFG.is_training: 
    if CFG.is_tuning:
        best_hps = tuner.get_best_hyperparameters()
        model = build_model_with_keras_tuner(best_hps[0])
    else:
        model = build_model_with_best_params()
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        "model.tf", 
        monitor="val_smape", 
        mode="min", 
        save_best_only=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=30,
        monitor="val_loss",
        mode="min",
        restore_best_weights=True
    )
    epochs = CFG.epochs
    learning_rate = model.optimizer.learning_rate.numpy()
    scheduler = get_cosine_decay_learning_rate_scheduler(epochs=epochs, lr_start=learning_rate, lr_end=learning_rate * 0.01)
    history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[checkpoints, early_stop, scheduler], verbose=2)
    model = tf.keras.models.load_model("model.tf", custom_objects={"smape": smape})
    models.append(model)
    pd.DataFrame(history.history).plot()
    plt.show()

previous_validation_dates = train.first_day_of_month.unique()[-(CFG.sequence_length + CFG.validation_length):-CFG.validation_length]
valid_tensors_list = []
for i in range(len(previous_validation_dates)):
    df = train[train.first_day_of_month == previous_validation_dates[i]]
    tensor = tf.reshape(tf.transpose(tf.constant([
            df.county_id,
            df.state_id,
            df.microbusiness_density
    ])), (len(df), 1, 3))
    valid_tensors_list.append(tensor)
smapes = []
state_smape_list = []
county_smape_list = []
date_smape_list = []
for model in models:
    print("=" * 50)
    valid_tensors = valid_tensors_list[0:CFG.sequence_length]
    train["prediction"] = 0.0
    dates = train.first_day_of_month.unique()[-CFG.validation_length:]
    for date in dates:
        X = tf.concat(valid_tensors[-CFG.sequence_length:], axis=1)
        prediction =model.predict(X)
        shape = (valid_tensors[0].shape[0], 1, 1)
        tensor = tf.concat([
            tf.reshape(valid_tensors[-1][:, :, 0], shape),   
            tf.reshape(valid_tensors[-1][:, :, 1], shape),   
            tf.reshape(prediction, shape)
        ], axis=-1)
        valid_tensors.append(tensor)
        train.loc[train.first_day_of_month == date, "prediction"] = prediction
    county_smapes = train[train.first_day_of_month >=dates[0]].groupby("cfips").apply(cauclate_smape)
    county_smapes = county_smapes.sort_values()
    mean_smape = np.mean(county_smapes)
#    print(county_smapes)
    county_smape_list.append(county_smapes)
    print(f"MEAN SMAPE:{mean_smape}")
    date_smapes = train[train.first_day_of_month >=dates[0]].groupby("first_day_of_month").apply(cauclate_smape)
    date_smapes = date_smapes.sort_values()
    date_smape_list.append(date_smapes)
#    print(date_smapes)
    print(f"Public LB:{np.mean(date_smapes[0])}")
    print(f"Private LB:{np.mean(date_smapes[1:4])}")
    state_smapes = train[train.first_day_of_month >=dates[0]].groupby("state").apply(cauclate_smape)
    state_smapes = state_smapes.sort_values()
    mean_state_smape = np.mean(state_smapes)
    print("state mape: ", mean_state_smape)
#    print(state_smapes)
    state_smape_list.append(state_smapes)
    smapes.append(mean_smape)
    # print(smapes)
smape_lstm = pd.DataFrame(state_smapes).iloc[:6].mean()
best_model = models[np.argmin(smapes)]
best_model.save("best_model.tf")
best_model_county_smapes = county_smape_list[np.argmin(smapes)]
final_time_here = time.time()
print("total time elapsed : ", final_time_here - start_time_here)
