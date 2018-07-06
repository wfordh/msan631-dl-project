import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

from mnist_data import train_dl, test_dl


class LogisticRegression(nn.Module):
	def __init__(self, input_size, num_classes):
		super(LogisticRegression, self).__init__() # why are we supering ourselves?
		self.linear = nn.Linear(input_size, num_classes)


	def forward(self, x):
		# not sure it matters which one we use
		out = F.log_softmax(self.linear(x), dim = 1) # is this what I want? or wrap it in F.log_softmax()? 
		#out = self.linear(x)
		return out

input_dim = 28
input_size = input_dim**2
n_classes = 10
n_epochs = 10
learning_rate = 0.01
path = Path('models/')

# Creating the model object
model = LogisticRegression(input_size, n_classes)

# Initiating loss and optimizer

def train_epochs(model, train_dl = train_dl, n_epochs = 10, lr = 0.01, wd = 0.0):
	parameters = filter(lambda p: p.requires_grad, model.parameters()) # is this necessary?
	optimizer = optim.SGD(parameters, lr = lr)
	model.train()
	for epoch in np.arange(n_epochs):
		for i, (x, y) in enumerate(train_dl):
			x = x.reshape(-1, input_size)
			out = model(x)
			loss = F.cross_entropy(out, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		p = path/'LR_model.pth'
		save_model(model, p)
		if epoch % 2 == 0:
			print('Epoch ' + str(epoch) + ' train loss: ' + str(loss.item()))
	test_loss(model, test_dl)


def test_loss(model, test_dl = test_dl):
	model.eval()
	total = 0
	correct = 0
	sum_loss = 0
	preds_list = list()

	for i, (x, y) in enumerate(test_dl):
		x = x.reshape(-1, input_size)
		batch = y.shape[0]
		out = model(x)
		loss = F.cross_entropy(out, y)
		preds = torch.argmax(out, dim = 1)
		sum_loss += batch*(loss.item())
		correct += preds.eq(y.data).sum().item()
		total += batch
		preds_list.append(preds)
		
	print("val loss and accuracy", sum_loss/total, correct/total)
	return preds_list

def save_model(m, p): 
	torch.save(m.state_dict(), p)

run_model = False
if run_model:
	train_epochs(model, train_dl)

