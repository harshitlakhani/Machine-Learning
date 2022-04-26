import os
from random import shuffle
import wandb
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


from model import SimpleNet
from dataset import CustomDataset

DATA_SRC_DIR = "/home/devi/Documents/PhD_Research/Dalhousie/courses/machine_learning_sem1/cs6505_course_project/"
LOG_DIR = "/home/devi/Documents/PhD_Research/Dalhousie/courses/machine_learning_sem1/cs6505_course_project/src/CNN/models/model_lr_0.0001_norm_martin_model/"

DEBUG = False

def save_models(model, log_dir, epoch):
    #TODO: Save checkpoint
    torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.model".format(epoch)))
    print("Checkpoint saved")

def test(model, batch_size, loss_fn, mode):
    model.eval()
    if mode == 'valid':
        dataset = CustomDataset(path = os.path.join(DATA_SRC_DIR, 'data/valid'))
    else:
        dataset = CustomDataset(path = os.path.join(DATA_SRC_DIR, 'data/test'))

    print("Number of {} images: {}".format(mode, len(dataset)))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_acc = 0.0
    test_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if DEBUG:
            import cv2
            img_ = images[0].numpy()
            img_ = img_ * dataset.stddev
            img_ += dataset.mean
            img_ *= 255
            img_ = img_.transpose(1, 0, 2)
            img_ = img_.transpose(0, 2, 1)
            print(images.shape)
            cv2.imwrite("{}_write.png".format(mode), img_)

        #Predict classes using images from the test set
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        test_loss += loss.cpu().data * images.size(0)

        _,prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)        


    #Compute the average acc and loss over all 900 test images
    test_acc = test_acc / len(dataset)
    test_loss = test_loss / len(dataset)

    return test_loss, test_acc

def train(training_dict):

    with wandb.init(project='test', entity='cs6505', config=training_dict): #mode="disabled"
        config = wandb.config
        dataset = CustomDataset(path = os.path.join(DATA_SRC_DIR, 'data/train'))
        print("Number of training images: {}".format(len(dataset)))

        if DEBUG:
            shuffle_ = False
        else:
            shuffle_ = True

        train_loader = DataLoader(dataset, config.batch_size, shuffle=shuffle_)

        num_classes = len(np.unique(dataset.labels))
        model = SimpleNet(num_classes)
        optimizer = Adam(model.parameters(), config['lr'])
        loss_fn = nn.CrossEntropyLoss()


        wandb.watch(model, loss_fn, log="all")

        min_loss = float('inf')

        for epoch in range(config.epochs):
            model.train()
            train_acc = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Clear all accumulated gradients
                optimizer.zero_grad()
                
                if DEBUG:
                    import cv2
                    img_ = images[0].numpy()
                    img_ = img_ * dataset.stddev
                    img_ += dataset.mean
                    img_ *= 255
                    img_ = img_.transpose(1, 0, 2)
                    img_ = img_.transpose(0, 2, 1)
                    print(images.shape)
                    cv2.imwrite("train_write.png", img_)


                outputs = model(images)
                # Compute the loss based on the predictions and actual labels
                loss = loss_fn(outputs, labels)

                if DEBUG:
                    print("Scores: {}".format(outputs.data))
                    print("Prediction: {}".format(torch.max(outputs.data, 1)[-1]))
                    print("Labels: {}".format(labels.data))
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.cpu().data * images.size(0) # what's with the multiplication term here?
                _, prediction = torch.max(outputs.data, 1)
                
                train_acc += torch.sum(prediction == labels.data)

            
            # Compute the average acc and loss over all 9000 training images
            train_acc = train_acc / len(dataset)
            train_loss = train_loss / len(dataset)

            # Evaluate on the validation set
            valid_loss, valid_acc = test(model, config.batch_size, loss_fn, 'valid')

            # Save the model if this epoch's loss is greater than previous loss values
            if train_loss < min_loss:
                save_models(model, config.log_dir, epoch)
                min_loss = train_loss

            # Log the metrics
            wandb.log({ 'Train loss': train_loss,
                        'Train Accuracy': train_acc,
                        'Validation loss': valid_loss,
                        'Validation Accuracy': valid_acc})
            print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Valid Accuracy: {}".format(epoch, train_acc, train_loss,valid_acc))
                        
                
if __name__ == '__main__':

    training_dict = {"lr": 0.0001,
                    "epochs": 500,
                    "batch_size": 64,
                    "log_dir": LOG_DIR}

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    train(training_dict)