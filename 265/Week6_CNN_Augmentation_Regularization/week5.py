import torch
import torchvision
import numpy as np
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import copy
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def data_augmentation():
    augmentations = transf.Compose([
        transf.RandomHorizontalFlip(),
        transf.RandomErasing(),
        # transf.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    ])
    return augmentations


def show_images(trainloader, classes):
    dataiter = iter(trainloader)
    images_b, labels_b = dataiter.next()
    for idx, image in enumerate(images_b):
        plt.imshow(images_b[idx].numpy().transpose(1, 2, 0))
        plt.title(classes[labels_b[idx]])
        plt.show()

def make_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5), #28x28
        nn.MaxPool2d(2, stride=2, padding=0), #14x14
        nn.Conv2d(12, 24, 3), #12x12
        nn.MaxPool2d(2, stride=2, padding=0), #6x6
        nn.Conv2d(24, 42, 3), #4x4
        nn.MaxPool2d(2, stride=2, padding=0), #2x2
        nn.Flatten(),
        nn.Linear(168, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model

def make_model_2():
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5), #28x28
        nn.MaxPool2d(2, stride=2, padding=0), #14x14
        nn.Conv2d(12, 24, 3), #12x12
        nn.MaxPool2d(2, stride=2, padding=0), #6x6
        nn.Conv2d(24, 42, 3), #4x4
        nn.MaxPool2d(2, stride=2, padding=0), #2x2
        nn.Flatten(),
        nn.Linear(168, 64),
        nn.ReLU(),
        # nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(32, 10)
    )
    return model

def main():
    images = torchvision.datasets.CIFAR10("CIFAR10", train=True, transform=transf.ToTensor(), download=False)
    train, val = torch.utils.data.random_split(images, [45000, 5000])


    train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1)
    show_images(train_loader, images.classes)
    model = make_model()
    adam = torch.optim.Adam(model.parameters())
    cross_entropy = nn.CrossEntropyLoss()
    model, train_acc, train_loss, val_acc, val_loss = train_model(model,cross_entropy, adam, 30, train_loader,
                                                                  val_loader, None)
    #torch.save(model, "Model_1")
    #np.save("train_acc_m1", train_acc)
    #np.save("val_acc_m1", val_acc)
    #np.save("train_loss_m1", train_loss)
    #np.save("val_loss_m1", val_loss)
    #train_acc = np.load("train_acc_m1.npy")
    #val_acc = np.load("val_acc_m1.npy")
    #train_loss = np.load("train_loss_m1.npy")
    #val_loss = np.load("val_loss_m1.npy")

    make_plots(train_acc, train_loss, val_acc, val_loss)

    model_2 = make_model_2()
    adam_2 = torch.optim.Adam(model_2.parameters(), weight_decay=0.001)
    model_2, train_acc_2, train_loss_2, val_acc_2, val_loss_2 = train_model(model_2, cross_entropy, adam_2, 30,
                                                                            train_loader, val_loader, data_augmentation())
    #torch.save(model, "Model_2")
    #np.save("train_acc_m2", train_acc_2)
    #np.save("val_acc_m2", val_acc_2)
    #np.save("train_loss_m2", train_loss_2)
    #np.save("val_loss_m2", val_loss_2)
    make_plots(train_acc_2, train_loss_2, val_acc_2, val_loss_2)



def make_plots(train_acc, train_loss, val_acc, val_loss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1,len(train_acc)+1),train_acc, label="Training")
    ax.plot(np.arange(1,len(val_acc)+1),val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training-validation accuracy")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(1,len(train_loss)+1), train_loss, label="Training")
    ax.plot(np.arange(1,len(val_loss)+1),val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training-validation loss")
    plt.legend()
    plt.show()

def train_model(model, loss_f, optimizer, num_epochs, train, val, augmentations):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_size = len(train.dataset)
    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_size = len(val.dataset)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        print("epoch",epoch+1)
        running_loss = 0.0
        running_corrects = 0
        torch.set_grad_enabled(True)
        for batch_idx, batch in enumerate(train):
            model.train()
            inputs, labels = batch
            if (augmentations != None):
                inputs = augmentations(inputs)
            inputs.to(device)
            labels.to(device)

            #Forward
            out = model(inputs)
            _, preds = torch.max(out, 1)

            #Compute objective function
            loss = loss_f(out, labels)

            #Clean the gradients
            model.zero_grad()

            #Accumulate partial deriviates wrt parameters
            loss.backward()

            #Step in the opposite direction og the gradient wrt optimizer
            optimizer.step()

            #stats
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss[epoch] = running_loss / train_size
        train_acc[epoch] = running_corrects / train_size
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss[epoch], train_acc[epoch]))
        torch.set_grad_enabled(False)
        running_loss = 0.0
        running_corrects = 0
        #torch.set_grad_enabled(False)
        for batch_idx, batch in enumerate(val):
            model.eval()
            inputs, labels = batch

            #Move to gpu if availible
            inputs.to(device)
            labels.to(device)

            #Forward
            out = model(inputs)
            _, preds = torch.max(out,1)

            #Compute objective function
            loss = loss_f(out, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_loss[epoch] = running_loss / val_size
        val_acc[epoch] = running_corrects / val_size

        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss[epoch], val_acc[epoch]))

        # deep copy the model
        if (val_acc[epoch] > best_acc):
            best_acc = val_acc[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, train_acc, train_loss, val_acc, val_loss


main()
