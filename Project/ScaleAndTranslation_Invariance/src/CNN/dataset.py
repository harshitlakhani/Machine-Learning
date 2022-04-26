import re
from tokenize import Double
import cv2
import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEBUG = False
class CustomDataset(Dataset):

    def __init__(self, path):
        imgs_path = path
        file_list = glob.glob(imgs_path + "*")
        # print(file_list)

        self.data = np.array([])
        for class_path in file_list:
            for imgs_path in glob.glob(class_path + "/*.png"):
                self.data = np.append(self.data, imgs_path)
        # print(self.data[:1])

        # regex search string (using groups), pipe denotes "or"
        # expression in paratheses denotes a regex group
        match = r"(Triangle|Square|Pentagon|Hexagon|Heptagon|Octagon|Nonagon|Circle|Star)"

        self.labels = np.array([])
        for label in self.data:
            class_name  = re.search(match, label) # finds match in filename
            self.labels = np.append(self.labels, class_name.group(1)) # returns match

        # values to use for shapes
        self.class_map = {"Triangle" : 0,
                          "Square"   : 1,
                          "Pentagon" : 2,
                          "Hexagon"  : 3,
                          "Heptagon" : 4,
                          "Octagon"  : 5,
                          "Nonagon"  : 6,
                          "Circle"   : 7,
                          "Star"     : 8}

        self.img_dim = (200, 200)

        # create a (number of examples, 2) numpy array
        self.images = np.stack((self.data, self.labels), axis = -1)

        self.transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            # transforms.CenterCrop((100, 100)),
                            # transforms.RandomCrop((80, 80)),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation(degrees=(-90, 90)),
                            # transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

        if DEBUG:
            self.images = np.expand_dims(self.images[0], axis=0)

    def __len__(self):
        # number of examples
        return self.images.shape[0]

    def __getitem__(self, idx):
        img_path, class_name = self.images[idx]

        img = cv2.imread(img_path)
        if DEBUG:
            self.idx = idx
            cv2.imwrite('cv2_write.png', img)
        
        class_id = self.class_map[class_name]

        if self.transforms is not None:
            img = self.transforms(img)

        class_id = torch.tensor(class_id)

        return torch.tensor(img, dtype=torch.float), class_id # convert to floats

def show_img(img):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(18,15))
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    dataset = CustomDataset(path = "/home/devi/Documents/PhD_Research/Dalhousie/courses/machine_learning_sem1/cs6505_course_project/data/test")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    data = iter(dataloader)
    images, labels = data.next()
    # show images
    show_img(torchvision.utils.make_grid(images))