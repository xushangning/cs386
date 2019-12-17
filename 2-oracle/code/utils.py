from keras.callbacks import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    print(cm.shape)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=np.arange(classes), yticklabels=np.arange(classes),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        if classes == 40 and (i + 1) % 4 == 0:
            plt.plot([i + 0.5, i + 0.5], [-0.5, 39.5], c='k', linewidth=2)
            plt.plot([-0.5, 39.5], [i + 0.5, i + 0.5], c='k', linewidth=2)
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.savefig(title, dpi=500)
    return ax


class VisualizationCallback(Callback):
    def __init__(self, on_batch=False):
        Callback.__init__(self)
        self.on_batch = on_batch

    def visualize(self):
        plt.figure(figsize=[12, 4])

        plt.subplot(1, 2, 1)
        plt.title("losses")
        plt.plot(self.loss_train, label='train loss')
        plt.plot(self.loss_val, label='val loss')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.plot(self.val_x, self.acc_train, label='train acc')
        plt.plot(self.val_x, self.acc_val, label='val acc')
        plt.grid()
        plt.legend()
        plt.show()

    def on_train_begin(self, logs={}):
        self.loss_train = []
        self.acc_train = []
        self.loss_val = []
        self.acc_val = []
        self.val_x = []

    def on_batch_end(self, batch, logs={}):
        if self.on_batch:
            self.loss_val.append(logs['val_loss'])
            self.acc_val.append(logs['val_acc'])
            print("Batch {0}: Train acc: {1:.4f}, Train loss: {2:.4f}".format(
                batch, logs['acc'], logs['loss']
            ))

    def on_epoch_end(self, epoch, logs={}):
        self.loss_val.append(logs['val_loss'])
        self.acc_val.append(logs['val_acc'])
        if not self.on_batch:
            self.loss_train.append(logs['loss'])
            self.acc_train.append(logs['acc'])
        self.val_x.append(len(self.acc_train) - 1)
        print("Epoch {0}: Train acc: {1:.4f}, Train loss: {2:.4f}, Val acc: {3:.4f}, Val loss: {4:.4f}".format(
            epoch, logs['acc'], logs['loss'], logs['val_acc'], logs['val_loss']
        ))

    def on_train_end(self, logs={}):
        plt.figure(figsize=[12, 4])

        plt.subplot(1, 2, 1)
        plt.title("losses")
        plt.plot(self.loss_train, label='train loss')
        plt.plot(self.loss_val, label='val loss')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("accuracy")
        plt.plot(self.acc_train, label='train acc')
        plt.plot(self.acc_val, label='val acc')
        plt.grid()
        plt.legend()

        plt.show()
