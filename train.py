import torch
from torch import nn
import torch.nn.functional as F
from NN import AttentionUNet
from torch.utils.data import DataLoader
from loader import BrainDataset
import os


#Getting data directory
train_images_inputs_dir = ["numpy_brain_data/train/inputs/" + item for item in  os.listdir("numpy_brain_data/train/inputs/")]
train_images_outputs_dir = ["numpy_brain_data/train/outputs/" + item for item in  os.listdir("numpy_brain_data/train/outputs/")]

#Creating Dataloader
training_dataset = BrainDataset(train_images_inputs_dir,train_images_outputs_dir)
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)

#Training model
device = torch.device("cuda")

model = AttentionUNet(in_channels=1, out_channels=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        
        images, labels = images.to(device), labels.to(device)

        outputs = model.forward(images)
        SR = model(images)
        SR_probs = F.sigmoid(SR)
        SR_flat = SR_probs.view(SR_probs.size(0),-1)
        GT_flat = labels.view(labels.size(0),-1).float()
        loss = criterion(SR_flat,GT_flat)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {running_loss/len(train_loader)}")

#Saving model
torch.save(model.state_dict(),"my_model.pt")