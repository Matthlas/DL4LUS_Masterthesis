import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Adapted from https://github.com/jannisborn/covid19_ultrasound

def evaluate_model(model, testX, testY, IL, batch_size, save_dir=None):
    predIdxs = model.predict(testX, batch_size=batch_size)
    # CSV: save predictions for inspection:
    df = pd.DataFrame(predIdxs) 
    df = pd.concat([df, pd.DataFrame(testY, columns=[f"test_{x}" for x in range(testY.shape[1])])], axis=1)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    if save_dir is not None:
        df.to_csv(os.path.join(save_dir, "_preds_last_epoch.csv"))

    num_classes = len(np.unique(testY))
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {IL.class_names}")
    print(f"Labels = {np.arange(0,num_classes)}")

    print('classification report sklearn:')
    print(
        classification_report(
            testY.argmax(axis=1), predIdxs, target_names=IL.class_names.astype(str), labels=np.arange(0,num_classes),
        )
    )

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    print('confusion matrix:')
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)


def plot_history(H, epochs, save_dir=None):
    # plot the training loss and accuracy
    N = len(H.history["loss"])
    if N != epochs:
        print(f"WARNING: History Length = {N} != epochs = {epochs}")
        print("Using History Length for plotting. Network might have stopped early.")
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, N), np.array(H.history['loss']), label='train_loss')
    plt.plot(np.arange(0, N), np.array(H.history['val_loss']), label='val_loss')
    plt.plot(np.arange(0, N), np.array(H.history['accuracy']), label='train_acc')
    plt.plot(np.arange(0, N), np.array(H.history['val_accuracy']), label='val_acc')
    plt.title('Training Loss and Accuracy on COVID-19 Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "training_history.png"))
        