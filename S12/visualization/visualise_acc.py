import matplotlib.pyplot as plt
import sys


##Plotting Accuracies and Losses

def plot_acc(test_accuracy,train_accuracy):
    try:

        fig, ax = plt.subplots(1, figsize=(15, 10))

        if test_accuracy is not None:
            ax.plot(test_accuracy, 'r')
        if train_accuracy is not None:
            ax.plot(train_accuracy, 'c')
        ax.legend(['Test Accuracy', 'Train Accuracy'], loc='best')
        ax.set_title('Accuracy for the models(Train vs Test)')

        fig.savefig('acc_trn_vs_tst.jpg')
        fig
        fig.show()
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(e).__name__ + " " + str(e))
        sys.exit(1)