# Adapted from https://github.com/jannisborn/covid19_ultrasound

print("Starting imports")
import argparse
import os
import numpy as np

print("Importing tf")
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
print("TF imports done.")
print("Importing custom modules")

from pocovidnet import MODEL_FACTORY
from pocovidnet.utils import Metrics


import sys; sys.path.insert(0, "../data/"); sys.path.insert(0, "../utils/")
from load_datasets import get_maastricht_loader, get_pocovid_loader, get_covid_us_loader, get_pocovid_covid_us_combined_dataset
from model_utils import evaluate_model, plot_history
print("Imported all packages successfully")

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument(
#     '-d', '--data_dir', required=True, help='path to input dataset'
# )
ap.add_argument('-ds', '--dataset', required=True, type=str, help="dataset to use. Available: maastricht, pocovid, covid_us, pocovid_covid_us_combined")
ap.add_argument('-pre', '--pretrained_model', required=True, type=str, help="Path to pretrained model to use. E.g. models/test_povocid_replicated_pocovid/fold_0/best_weights")
ap.add_argument('-pre_ds', '--pretraining_ds', required=True, type=str, help="Name of dataset used for previous training. Available: maastricht, pocovid, covid_us, pocovid_covid_us_combined")

ap.add_argument('-m', '--model_dir', type=str, default='models/')
ap.add_argument('-f', '--fold', type=int, default='0', help='fold to take as test data')
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=30)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-t', '--trainable_base_layers', type=int, default=1)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
ap.add_argument('-id', '--model_id', type=str, default='vgg_base')
ap.add_argument('-ls', '--log_softmax', type=bool, default=False)
ap.add_argument('-n', '--model_name', type=str, default='test')
ap.add_argument('-hs', '--hidden_size', type=int, default=64)
ap.add_argument('-s', '--stride', type=int, default=5)
args = vars(ap.parse_args())
print("Working directory:", os.getcwd())

# Initialize hyperparameters
# DATA_DIR = args['data_dir']
DATASET = args['dataset']
PRETRAINED_MODEL_PATH = args['pretrained_model']
PRETRAINED_MODEL_PATH = os.path.join(PRETRAINED_MODEL_PATH, f"fold_{args['fold']}", "best_weights")
PRETRAINING_DS = args['pretraining_ds']
MODEL_NAME = args['model_name'] + "_" + DATASET
FOLD = args['fold']
MODEL_DIR = os.path.join(args['model_dir'], MODEL_NAME, f'fold_{FOLD}')
LR = args['learning_rate']
EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
MODEL_ID = args['model_id']
TRAINABLE_BASE_LAYERS = args['trainable_base_layers']
IMG_WIDTH, IMG_HEIGHT = args['img_width'], args['img_height']
LOG_SOFTMAX = args['log_softmax']
HIDDEN_SIZE = args['hidden_size']
STRIDE = args['stride']

# Check if model class exists
if MODEL_ID not in MODEL_FACTORY.keys():
    raise ValueError(
        f'Model {MODEL_ID} not implemented. Choose from {MODEL_FACTORY.keys()}'
    )

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
print(f'Model parameters: {args}')

print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}...")
model_pre = keras.models.load_model(PRETRAINED_MODEL_PATH)
print("Successfully loaded pretrained model.")

print(f'Loading pre training: {PRETRAINING_DS} ds test data...')
if PRETRAINING_DS == "pocovid":
    IL_pre = get_pocovid_loader()
elif PRETRAINING_DS == "covid_us":
    IL_pre = get_covid_us_loader()
elif PRETRAINING_DS == "pocovid_covid_us_combined":
    IL_pre = get_pocovid_covid_us_combined_dataset()
else:
    raise ValueError(f"Dataset {PRETRAINING_DS} not implemented as pretrained ds. Choose from pocovid, covid_us, pocovid_covid_us_combined")

IL_pre.stride(STRIDE)
train_pre, test_pre = IL_pre.get_tf_dataset(FOLD)
testX_pre, testY_pre = tf.data.Dataset.get_single_element(test_pre.batch(len(test_pre)))
testX_pre = testX_pre.numpy()
testY_pre = testY_pre.numpy()
num_classes = len(np.unique(testY_pre))

# make predictions on the testing set
print('Evaluating pretrained network on pretrained ds...')
evaluate_model(model_pre, testX_pre, testY_pre, IL_pre, batch_size=BATCH_SIZE)

print(f'Create dataloader for {DATASET} ds...')
if DATASET == "maastricht":
    IL = get_maastricht_loader()
elif DATASET == "pocovid":
    IL = get_pocovid_loader()
elif DATASET == "covid_us":
    IL = get_covid_us_loader()
elif DATASET == "pocovid_covid_us_combined":
    IL = get_pocovid_covid_us_combined_dataset()
else:
    raise ValueError(f"Dataset {DATASET} not implemented. Choose from maastricht, pocovid, covid_us, pocovid_covid_us_combined")

IL.stride(STRIDE)
train, test = IL.get_tf_dataset(FOLD)


print(
    f'\nNumber of training samples: {len(train)} \n'
    f'Number of testing samples: {len(test)}'
)

print("Loading Dataset into memory...")

trainX, trainY = tf.data.Dataset.get_single_element(train.batch(len(train)))
testX, testY = tf.data.Dataset.get_single_element(test.batch(len(test)))

trainX = trainX.numpy()
trainY = trainY.numpy()
testX = testX.numpy()
testY = testY.numpy()

num_classes = len(np.unique(trainY))
print("Done.")

# Evaluate pretrained model on maastricht dataset
print(f'Evaluating pretrained network on {DATASET} ds before continued training...')
evaluate_model(model_pre, testX, testY, IL, batch_size=BATCH_SIZE)


# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=10,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Load the VGG16 network
# model = MODEL_FACTORY[MODEL_ID](
#     input_size=(IMG_WIDTH, IMG_HEIGHT, 3),
#     num_classes=num_classes,
#     trainable_layers=TRAINABLE_BASE_LAYERS,
#     log_softmax=LOG_SOFTMAX,
#     hidden_size=HIDDEN_SIZE
# )

# Define callbacks
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

mcp_save = ModelCheckpoint(
    # os.path.join(MODEL_DIR, 'fold_' + str(FOLD) + '_epoch_{epoch:02d}'),
    os.path.join(MODEL_DIR, 'best_weights'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=7,
    verbose=1,
    epsilon=1e-4,
    mode='min'
)
# To show balanced accuracy
metrics = Metrics((testX, testY), model_pre)

# train the head of the network
print('Continue training pretrained model...')
H = model_pre.fit(
    trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    # steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    # validation_steps=len(testX) // BATCH_SIZE,
    epochs=20,
    callbacks=[earlyStopping, mcp_save, reduce_lr_loss, metrics]
)

# make predictions on the testing set
print('Evaluating network...')
evaluate_model(model_pre, testX, testY, IL, batch_size=BATCH_SIZE, save_dir=MODEL_DIR)

# plot the training loss and accuracy
plot_history(H, EPOCHS, save_dir=MODEL_DIR)

print('Done, shuttting down!')