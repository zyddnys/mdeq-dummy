import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models_cifar10 import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

def eval(net) :
	correct = 0
	total = 0
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images, labels = images.cuda(), labels.cuda()
			# calculate outputs by running images through the network
			outputs = net(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
	return correct / total

def train(net: MDEQModelCifar10) :
	criterion = nn.CrossEntropyLoss()
	writer = SummaryWriter('cifar10_summary/')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	train_step = 0
	for epoch in range(20):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.cuda(), labels.cuda()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs, train_step, writer = writer)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 100 == 99:    # print every 100 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 100))
				writer.add_scalar('loss', running_loss, train_step)
				running_loss = 0.0
			train_step += 1
		ea = eval(net)
		writer.add_scalar('eval_acc', ea, train_step)
	print('Finished Training')


def main() :
	cfg = {
		'num_layers': 4,
		'f_thres': 24,
		'b_thres': 24,
		'pretrain_steps' : 100,
	}
	f = ModelF(cfg).cuda()
	inject = ModelInjection(cfg).cuda()
	net = MDEQModelCifar10(cfg, f, inject).cuda()
	train(net)

if __name__ == '__main__' :
	main()
