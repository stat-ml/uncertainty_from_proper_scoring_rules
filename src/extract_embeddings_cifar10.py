import sys
sys.path.insert(0, './')
import torch
from external_repos.pytorch_cifar10.utils import get_dataloaders, get_model

####
architecture = 'resnet18' 
model_id = 0
loss_function = "cross_entropy"
####


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data
trainloader, testloader = get_dataloaders()


print(f'Using {architecture} for evaluation...')
# Model
print('==> Building model..')

net = get_model(architecture=architecture)

print("Used device is ", device)
net = net.to(device)
#

def extract_embeddins():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))




print("Success!")
