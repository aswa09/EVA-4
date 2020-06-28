import matplotlib.pyplot as plt
import sys

##Plotting Accuracies and Losses

def plot_acc_loss(test_losses, test_acc,train_losses, train_acc):
	try:
  
		fig,ax1 = plt.subplots(1, figsize=(15,10))
		fig.suptitle('Loss and Accuracy for the models(Train vs Test)', fontsize=16)
		ax1.plot(test_losses,'b')
		ax1.plot(train_losses,'k')
		ax1.legend(['Test Loss','Train Loss'], loc='best')
		#ax1.set_title("Train vs Test Loss")

		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		
		ax2.plot(test_acc,'r')
		ax2.plot(train_acc,'c')
		ax2.legend(['Test Accuracy','Train Accuracy'], loc='best')
		#ax2.set_title("Train vs Test Accuracy")

		fig.savefig('acc_vs_loss.jpg')
		fig
		fig.show()
	except Exception as e:
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(e).__name__ + " " + str(e))
		sys.exit(1)