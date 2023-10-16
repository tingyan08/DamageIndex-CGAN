import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

def dataloader(args, dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    data_loader = DataLoader(DamageIndex("./Plane_data_750x800", transform=transform, continuous_column=args.continuous_column, discrete_column=args.discrete_column), batch_size=batch_size, shuffle=True)
    if not os.path.exists(os.path.join(args.result_dir, "model")):
        os.makedirs(os.path.join(args.result_dir, "model"))
    args.dis_attribute = data_loader.dataset.dis_attribute
    args.con_attribute = data_loader.dataset.con_attribute

    # Save args
    with open(os.path.join(args.result_dir, "model", "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return data_loader

class DamageIndex(Dataset):
    """Dataset class for the DamageIndex dataset."""

    def __init__(self, root="./DI", transform=None, img_suffix='.JPG', continuous_column = [], 
                                    discrete_column=["AR", "HR", "VR", "DI"]):
        """Initialize and preprocess the DamageIndex dataset."""
        self.image_dir = os.path.join(root, "images")
        self.df = pd.read_csv(os.path.join(root, "damageindex(even).csv"))
        self.continuous_column = continuous_column
        self.discrete_column = discrete_column
        self.con_label = self.df[self.continuous_column]
        self.con_attribute = self.con_label.columns.to_list()
        self.dis_label = self.df[self.discrete_column]
        self.preprocess()
        self.data = self.df.iloc[:,0].values
        self.transform = transform
        self.img_suffix = img_suffix
        

    def preprocess(self):
        # Count number of tasks and unique value for each task
        self.n_each_task = []
        for dl in self.discrete_column:
            self.n_each_task.append(self.dis_label[dl].nunique())

        # One hot for discrete label
        self.dis_label = self.dis_label.applymap(str)
        if len(self.discrete_column) != 0:
            self.dis_label = pd.get_dummies(self.dis_label)
        
        self.dis_attribute = self.dis_label.columns.to_list()

        self.dis_label = self.dis_label.values
        self.con_label = self.con_label.values

        print(f"Length of discrete label : {len(self.dis_attribute)} ; Length of continuous label : {len(self.con_attribute)}")
        print('Finished preprocessing the DamageIndex dataset...')

    # def preprocess(self):
    # """Preprocess the DamageIndex attribute file."""
    # for i, attr_name in enumerate(self.df.columns[:-1]):
    #     self.attr2idx[attr_name] = i
    #     self.idx2attr[i] = attr_name
        

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, dis_label, con_label = self.data[index], self.dis_label[index], self.con_label[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(dis_label), torch.FloatTensor(con_label)

    def __len__(self):
        """Return the number of images."""
        return len(self.data)

if __name__ == "__main__":
    input_size = 128
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = DamageIndex('./Plane_data_750x800', transform=transform, continuous_column = ["DI"], discrete_column=["AR", "HR", "VR"])
    print(dataset[100])