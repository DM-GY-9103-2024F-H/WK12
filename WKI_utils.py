import torch

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data_utils import LFWUtils as LFWUtils_Linear


class FaceDataset(Dataset):
  @staticmethod
  def toTensor(pxs):
    return pxs.reshape(-1, LFWUtils.IMAGE_SIZE[1], LFWUtils.IMAGE_SIZE[0])

  @staticmethod
  def toPixels(img):
    return img.reshape(-1)

  def __init__(self, imgs, labels, transform=None):
    self.imgs = imgs
    self.labels = labels
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img, label = self.imgs[idx], self.labels[idx]
    if self.transform and torch.is_grad_enabled():
      img = FaceDataset.toPixels(self.transform(FaceDataset.toTensor(img)))
    return img, label


class LFWUtils(LFWUtils_Linear):
  @staticmethod
  def train_test_split(test_pct=0.5, random_state=101010, return_loader=True, train_transform=None, test_transform=None):
    train, test = LFWUtils_Linear.train_test_split(test_pct=test_pct, random_state=random_state)

    if return_loader:
      x_train = Tensor(train["pixels"])
      y_train = Tensor(train["labels"]).long()

      x_test = Tensor(test["pixels"])
      y_test = Tensor(test["labels"]).long()

      train_dataloader = DataLoader(FaceDataset(x_train, y_train, train_transform), batch_size=256, shuffle=True)
      test_dataloader = DataLoader(FaceDataset(x_test, y_test, test_transform), batch_size=512)

      return train_dataloader, test_dataloader
    else:
      return train, test

  @staticmethod
  def get_labels(model, dataloader):
    model.eval()
    with torch.no_grad():
      data_labels = []
      pred_labels = []
      for x, y in dataloader:
        y_pred = model(x).argmax(dim=1)
        data_labels += [l.item() for l in y]
        pred_labels += [l.item() for l in y_pred]
      return data_labels, pred_labels
