from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """Train the model over the chosen layers

    This method, given a pretrained "Inception v3" model,
    applies a fine tuning of the last layer on a new dataset
    indexed by the dataloader.

    Parameters
    ----------
    model :
        Pre trained model
    dataloaders :
        Data Loader of the training and validation set.
    criterion :
        Loss Criterion such as Cross Entropy loss.
    optimizer :
        Optimizer chosen to train the model.
    num_epochs : int
        Number of epochs over which train the model, default
        25 epochs.

    Returns
    -------
    model :
        Instance of the trained model
    val_acc_history :
        Trend of the validation accuracy

    """

    # Initializing the model
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Iterating over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

                # Reset values for each phase
            running_loss = 0.0
            running_corrects = 0

            # Iterating over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy of the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def data_transformation(data_dir, batch_size):
    """Adapt new dataset to the pretrained network

    Parameters
    ----------
    data_dir :
        Directory of the folder with the new
        data set
    batch_size :
        Dimension of the batch

    Returns
    -------
    dataloaders_dict :
        Dataloaders dictionary
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size_hyp),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size_hyp),
            transforms.CenterCrop(input_size_hyp),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}
    return dataloaders_dict


def initialize_model(num_classes):
    """Initialize the model by setting the number
    of classes of the new data set and initially
    freezing all the parameters.

    Parameters
    ----------
    num_classes :
        Number of classes related to the
        new dataset.

    Returns
    -------
    model :
        Instance of the Inceptionv3 model that will be
        trained over the new data set.
    """
    model = torchvision.models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features_1 = model.fc.in_features
    num_features_2 = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_features_2, num_classes)
    model.fc = nn.Linear(num_features_1, num_classes)
    return model


def setting_parameters(model, learning_rate, momentum):
    """Create a list with the all the parameters
    that will be updated during training and
    the SGD optimizer.

    Parameters
    ----------
    model :
        Directory of the folder with the new
        data set
    learning_rate :
        Step size of the loss fcn.
    momentum :
        Momentum related to the optimization.

    Returns
    -------
    optimizer_ft :
        Optimizer instance
    """
    model = model.to(device)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    return optimizer_ft


# parameters
num_classes_hyp = 20
learning_rate_hyp = 1e-3
criterion_hyp = nn.CrossEntropyLoss()
momentum_hyp = 0.9
batch_size_hyp = 8
num_epochs_hyp = 3
input_size_hyp = 299
data_dir_hyp = "C:/Users/borti/Desktop/UniCa/Magistrale/1st Year/2nd Sem/Machine learning/Project/cropped/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initializing the model
model_ft = initialize_model(num_classes_hyp)
# Setting the parameters
optimizer_aux = setting_parameters(model_ft, learning_rate_hyp, momentum_hyp)
# Transforming the data
data_loaders_dict_aux = data_transformation(data_dir_hyp, batch_size_hyp)
# Training the selected layer
model_ft, hist = train_model(model_ft, data_loaders_dict_aux, criterion_hyp, optimizer_aux, num_epochs=num_epochs_hyp)
# Saving the model to be used in the webcam
torch.save(copy.deepcopy(model_ft.state_dict()), 'iCubeInceptionv3')
