from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import os


class DataLoad(Dataset):
    """Dataset load
    Args:
        root: Path to the dataset file
        mode: Type of data set(train, test)
        imgsize: Sample size
    """
    
    def __init__(self, root, mode, image_size=[102,160,160]):
        super(DataLoad, self).__init__()
        self.root = root
        self.mode = mode
        self.image_size = image_size

        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root, "train", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "train", "LRHS_Elastic1000"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            self.HRMS = os.listdir(os.path.join(self.root, "train", "hrMS"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))

        elif self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root, "test", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "test", "LRHS_Elastic1000"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            self.HRMS = os.listdir(os.path.join(self.root, "test", "hrMS"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):
        gt_HS, LR_HS, HR_MS = self.gtHS[index], self.LRHS[index], self.HRMS[index]
        data_ref = loadmat(os.path.join(self.root, self.mode, "gtHS", gt_HS))['gtHS'] # size: C, H, W
        data_LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS_Elastic1000", LR_HS))['LRHS'] # C, h ,w 
        data_HRMS = loadmat(os.path.join(self.root, self.mode, "hrMS", HR_MS))['hrMS'] # c, H, W
        return data_HRMS, data_LRHS, data_ref


