from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
import numpy as np
import os
import time


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#parser.add_argument('--batch-size', type=int, default=256, metavar='N',
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
#parser.add_argument('--epochs', type=int, default=50, metavar='N',
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
#parser.add_argument('--lr', type=float, default=512/1300, metavar='LR',
parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                    help='weight decay (default: 0.01)')
parser.add_argument('--corrupt-prob', type=float, default=0.0, metavar='P',
                    help='label corrupt probability (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp-name', type=str, default='', help='experimenal name (default: '')')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.seed = int(time.time()*1000)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class MNISTRandomLabels(datasets.MNIST):
    """MNIST dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corruptprob=0.0, num_classes=10, **kwargs):
        super(MNISTRandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corruptprob > 0:
            self.corrupt_labels(corruptprob)

    def corrupt_labels(self, corruptprob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corruptprob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels


dataset_size = 60000 # only sample the first 10000 samples
test_dataset_size = 10000
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader_aug = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(
        [MNISTRandomLabels(root='../../data', train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]),
        corruptprob=args.corrupt_prob, num_classes=10),
        MNISTRandomLabels(root='../../data', train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.RandomRotation((-20, 20))
                   ]),
        corruptprob=args.corrupt_prob, num_classes=10)]), 
    batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))), shuffle=False, **kwargs)

train_loader = torch.utils.data.DataLoader(
    MNISTRandomLabels(root='../../data', train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]),
    corruptprob=args.corrupt_prob, num_classes=10), 
    batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))), shuffle=False, **kwargs) 

full_train_loader = torch.utils.data.DataLoader(
    MNISTRandomLabels(root='../../data', train=True, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), 
    num_classes=10, corruptprob=args.corrupt_prob),
    batch_size=dataset_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))), shuffle=False, **kwargs) 
    

test_loader = torch.utils.data.DataLoader(
    MNISTRandomLabels(root='../../data', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]),
    num_classes=10, corruptprob=args.corrupt_prob),
    batch_size=args.test_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(test_dataset_size))), shuffle=False, **kwargs)

# train_dataset = datasets.MNIST('../../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))

# wholedataset=torch.utils.data.ConcatDataset([datasets.MNIST('../../data', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
#                   ])), datasets.MNIST('../../data', train=False, transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
#                   ]))])


# wholeset_batches = torch.utils.data.DataLoader(wholedataset, batch_size=1, shuffle=False)

# new data loader (use larger batch-sze) for evaluate the loss of whole training data and whole test data along the training path
#evaluate_batch_size = dataset_size // 2 # can use other values for efficiency
#eval_train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../../data', train=True, download=True,
#                   transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
#                   ])),
#    batch_size=evaluate_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(dataset_size))), shuffle=False, **kwargs) 
#eval_test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
#                       transforms.ToTensor(),
#                       transforms.Normalize((0.1307,), (0.3081,))
#                   ])),
#    batch_size=evaluate_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(test_dataset_size))), shuffle=False, **kwargs)


# def resampler(sampler, batch_size=1, drop_last=False):
#     batch = []
#     for idx in sampler:
#         batch.append(idx)
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#     if len(batch) > 0 and not drop_last:
#         yield batch

# resample_batches = list(resampler(torch.randperm(args.batch_size), batch_size=resample_batch_size))
# print(resample_batches)
# # trainbatch_resampler = torch.utils.data.sampler.BatchSampler(torch.randperm(args.batch_size), batch_size=resample_batch_size, drop_last=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(784,320, bias=False)
        self.fc2 = nn.Linear(320, 50, bias=False)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def initialize_model(model_to_init):
    init_stdv = 1. # control the norm of initial weight
    total_norm = 0
    for p in model_to_init.parameters():
        if p.data is not None:
            p.data *= init_stdv
            total_norm += p.data.norm() ** 2
    print("initialization succeed, total norm is", math.sqrt(total_norm))


criterion = nn.CrossEntropyLoss()
criterion_no_average = nn.CrossEntropyLoss(size_average=False)

use_new_init = True
if not use_new_init:
    # Load checkpoint.
    print('==> Resuming from initialpoint..')
    assert os.path.isdir('initialpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./initialpoint/initialmodel.t7')
    model = checkpoint['model']
else:
    model=Net()
    initialize_model(model) # fix initialization
    print('Saving initial model..')
    state = {
        'model': model,
    }
    if not os.path.isdir('initialpoint'):
        os.mkdir('initialpoint')
    torch.save(state, './initialpoint/initialmodel.t7')

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

tmp_model = Net()
if args.cuda:
    tmp_model.cuda()
tmp_model.load_state_dict(model.state_dict())
optimizer_tmp = optim.SGD(tmp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# for calculate the gradient for train dataset

def train(epoch):
    #GE1 = []
    #GE2 = []
    var_gradient_array, empirical_loss_record_array, population_loss_record_array, GE_array = np.array([]), np.array([]), np.array([]), np.array([])
    len_batches = len(train_loader)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_aug):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
		
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
    
        if (epoch == args.epochs) and (len_batches-batch_idx <= 50):
            for batch_idx1, (data1, target1) in enumerate(full_train_loader):
                if batch_idx1 == 1:
                    break
                if args.cuda:
                    data1, target1 = data1.cuda(), target1.cuda()
                data1, target1 = Variable(data1), Variable(target1)
                optimizer.zero_grad()
		
                output1 = model(data1)
                loss1 = criterion(output1, target1)
                loss1.backward()
                
            tmp_model.load_state_dict(model.state_dict())
            tmp_var = 0                
            resample_batch_size = args.batch_size # which is m
            for idx, (tmp_data, tmp_target) in enumerate(iterate_minibatches(data1.data, target1.data, batchsize=resample_batch_size, shuffle=False)):
                 tmp_data, tmp_target = Variable(tmp_data), Variable(tmp_target)
                 optimizer_tmp.zero_grad()
                 one_batch_variance = 0
                 tmp_output = tmp_model(tmp_data)
                 tmploss = criterion(tmp_output, tmp_target)
                 tmploss.backward()
                 for p, tmp_p in zip(model.parameters(), tmp_model.parameters()):
                    one_batch_variance += ((p.grad.data-tmp_p.grad.data) ** 2).sum()
                    tmp_var += one_batch_variance
            tmp_var /= idx
            var_gradient = tmp_var.cpu()
        
        #calculate the empirical gradient variance
        #step1: sample another full dataset
        #step2: compute full grad p.grad.data at current iterate
        #step3: sample a batch of data from the full dataset, compute the gradient tmp_p.grad.data
        #step4: (p.grad.data-tmp_p.grad.data) ** 2
        
        
##            tmp_model.load_state_dict(model.state_dict())
##            tmp_var = 0
##            resample_batch_size = args.batch_size // 10 # which is m
##            for idx, (tmp_data, tmp_target) in enumerate(iterate_minibatches(data.data, target.data, batchsize=resample_batch_size, shuffle=False)):
##                 tmp_data, tmp_target = Variable(tmp_data), Variable(tmp_target)
##                 optimizer_tmp.zero_grad()
##                 one_batch_variance = 0
##                 tmp_output = tmp_model(tmp_data)
##                 tmploss = criterion(tmp_output, tmp_target)
##                 tmploss.backward()
##                 for p, tmp_p in zip(model.parameters(), tmp_model.parameters()):
##                    one_batch_variance += ((p.grad.data-tmp_p.grad.data) ** 2).sum()
##                    tmp_var += one_batch_variance
##            tmp_var /= idx
##            var_gradient.append(tmp_var)
            
            # calculate the generalization error
            # compute the training loss on the all training set
            train_total_loss = 0
            for idx, (tmp_data, tmp_target) in enumerate(train_loader):
                if args.cuda:
                    tmp_data, tmp_target = tmp_data.cuda(), tmp_target.cuda()
                tmp_data, tmp_target = Variable(tmp_data), Variable(tmp_target)
                tmp_output = model(tmp_data)
                train_total_loss += criterion_no_average(tmp_output, tmp_target)
    
            # now the training loss at w_t is train_total_loss/dataset_size
            empirical_loss_record = train_total_loss.cpu().data.item()/dataset_size
    
            test_total_loss = 0
            for idx, (tmp_data, tmp_target) in enumerate(test_loader):
                if args.cuda:
                    tmp_data, tmp_target = tmp_data.cuda(), tmp_target.cuda()
                tmp_data, tmp_target = Variable(tmp_data), Variable(tmp_target)
                tmp_output = model(tmp_data)
                test_total_loss += criterion_no_average(tmp_output, tmp_target)
            population_loss_record = test_total_loss.cpu().data.item()/test_dataset_size
    
            # now the population loss is (train_total_loss + test_total_loss)/(dataset_size + test_dataset_size)
            #population_loss_record.append((train_total_loss.data.item() + test_total_loss.data.item())/(dataset_size + test_dataset_size))
            GE = np.abs(np.subtract(empirical_loss_record, population_loss_record))
            #loss_diff = np.abs(empirical_loss_record[-1] - population_loss_record[-1])
            #GE1.append(loss_diff)
            print("Iteration {} information: \n gradient variance {} \n empirical loss {} \n population loss {} \n generalization error {}".format(batch_idx, var_gradient, empirical_loss_record, population_loss_record, GE))

            var_gradient_array = np.append(var_gradient_array, var_gradient)
            empirical_loss_record_array = np.append(empirical_loss_record_array, empirical_loss_record)
            population_loss_record_array = np.append(population_loss_record_array, population_loss_record)
            GE_array = np.append(GE_array, GE)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), dataset_size,
                100. * batch_idx / len(train_loader), loss.data.item()))
    #print("Epoch{} information \n empirical loss {} \n population loss {} \n loss diff {}".format(epoch, empirical_loss_record[-1], population_loss_record[-1], GE1[-1]))
    # save files
    exp_dir = os.path.join('./saved_files', 'mnist_'+str(args.corrupt_prob)+'_'+str(args.lr)+args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_fn_1 = os.path.join(exp_dir, "var_gradient.txt")
    log_fn_2 = os.path.join(exp_dir, "GE.txt")
    np.savetxt(log_fn_1, var_gradient_array, fmt='%6.4f')
    np.savetxt(log_fn_2, GE_array, fmt='%6.4f')
    #print("gradient variance are ", var_gradient_array)
    #print("GE are ", GE_array)


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item() # sum up batch loss
        pred = torch.max(output.data, dim=1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # test_loss /= 1
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_dataset_size,
        100. * correct / test_dataset_size))


for epoch in range(1, args.epochs + 1):
    train(epoch)
#    test()
#    total_norm = 0
#    i=0
#    for p in model.parameters():
#        total_norm += p.data.norm() ** 2
#	
#    print("\n now total_norm is", math.sqrt(total_norm))

#print("GE1 are ", GE1[-1])
#print("empirical losses are ", empirical_loss_record[-1])
#print("population losses are ", population_loss_record[-1])
