from keras.callbacks import *
import matplotlib.pyplot as plt


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
        self.val_x.append(len(self.acc_train)-1)
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
