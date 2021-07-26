import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

from one_hot_encoding import encode


class MyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_file_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, index) -> T_co:
        image_root = self.image_file_paths[index]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = image_name.split("-")[-1].replace(".jpg", "")
        return image, encode(label)


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def get_train_data_loader():
    dataset = MyDataset(os.path.join(os.path.dirname(__file__), "data/train"), transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = MyDataset(os.path.join(os.path.dirname(__file__), "data/test"), transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_predict_data_loader():
    dataset = MyDataset(os.path.join(os.path.dirname(__file__), "data/predict"), transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)
