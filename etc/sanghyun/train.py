import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from model.models import *

from tqdm import tqdm

print("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:[%s]."%(device))

#### Dataset ####
resize_w = 226
resize_h = 226
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                            transforms.Resize((resize_w, resize_h)),
                            ])
train_set = torchvision.datasets.ImageFolder(root = "./train/new_images",
                                           transform = trans)
test_set = torchvision.datasets.ImageFolder(root = "./train/new_valid",
                                           transform = trans)

print(train_set.__len__())

print("Dataset Done.")

#################
#### Data Iterator ####
BATCH_SIZE = 128
train_iter = torch.utils.data.DataLoader(train_set,
                         batch_size = BATCH_SIZE,
                         shuffle = True,
                         num_workers = 0)
#test_iter = torch.utils.data.DataLoader(test_set,
#                                        batch_size = BATCH_SIZE,
#                                        shuffle = True,
#                                        num_workers = 0)
print("Data Iterator Done.")
#######################

num_classes = 18
model_name = "ViT"
model = ModelList.parse_model(model_name)(num_classes).to(device)
print(model)
#weight = torch.tensor([1.,1.,3.]*6).cuda()
#loss = nn.CrossEntropyLoss(weight=weight)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(model.parameters(), lr=1e-3)
print("Define Model Done.")
####################

#### Evaluation Function ####
def func_eval(model, data_iter, device):
    with torch.no_grad():
        n_total, n_correct = 0, 0
        model.eval()
        for batch_in, batch_out in tqdm(data_iter):
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1, 3, resize_w, resize_h).to(device))
            _, y_pred = torch.max(model_pred.data, 1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train()
    return val_accr
print("Evaluation Function Done.")

print("Start training.")
#model.load_state_dict(torch.load("./saved/1model_new.pt"))
model.train()
EPOCHS, print_every = 1000, 3
for epoch in range(EPOCHS):
    loss_val_sum = 0
    f1_val_sum= 0
    for batch_in, batch_out in tqdm(train_iter):
        y_pred = model.forward(batch_in.view(-1, 3, resize_w, resize_h).to(device))
        loss_out = loss(y_pred, batch_out.to(device))
        optm.zero_grad()
        loss_out.backward()
        optm.step()
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(model,train_iter,device)
        #test_accr = func_eval(model, test_iter, device)
        #print("epoch:[%d] loss:[%.6f] train_accr:[%.6f] test_accr:[%.6f]."%(epoch, loss_val_avg, train_accr, test_accr))
        print("epoch:[%d] loss:[%.6f] train_accr:[%.6f]."%(epoch, loss_val_avg, train_accr))
        torch.save(model.state_dict(), "./saved/"+str(epoch)+"model_new.pt")
print("training Done.")
