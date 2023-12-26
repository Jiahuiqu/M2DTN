from torch.utils.data import DataLoader
from DataLoadpavia import DataLoad
from torch import nn, optim
import torch
from metrics_utils import *
import numpy as np
from model import Backbone
import random
import os

# Training condition settings
device = 'cuda:0'
batch_size = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(model):
    def init_func(m):
        classname = m.__class__.__name__
        # hasattr Determining whether an object has a particular property
        if hasattr(m, 'weight'):
            torch.nn.init.normal_(m.weight.data, mean=0, std=1)

    model.apply(init_func)


if __name__ == "__main__":

    set_seed(1)

    Run_loss, Test_loss = np.array([]), np.array([])
    train_data = DataLoad("pavia", mode="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    test_data = DataLoad("Pavia", mode="test")
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)

    criteon = nn.L1Loss()

    model = Backbone(102, 4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    """
    state = torch.load('bestmodel.pth')
    model.load_state_dict(state['model'])
    best_cc = state['best_cc']
    print('best_cc', best_cc)
    del state
    print('model finish')
    """

    for i in range(1500):
        pn, runloss, cc, sam = 0, 0, 0, 0
        for step, (hrMS, lrHS, ref) in enumerate(train_loader):
            hrMS = hrMS.type(torch.float).to(device)
            lrHS = lrHS.type(torch.float).to(device)
            ref = ref.type(torch.float).to(device)
            model.train()
            output = model(lrHS, hrMS)
            running_loss = criteon(output, ref)
            pn += psnr(output.cpu().detach().numpy(), ref.cpu().detach().numpy())
            cc += CC_function1(output.cpu().detach().numpy(), ref.cpu().detach().numpy())
            sam += SAM(output.cpu().detach().numpy(), ref.cpu().detach().numpy())            
            optimizer.zero_grad()
            # backward
            running_loss.backward()
            optimizer.step()
            runloss += running_loss.item()
        print('\nepoch', i + 1, 'train_loss', runloss / (step + 1), 'psnr', pn / (step + 1), 'CC', cc / (step + 1),
              'SAM', sam / (step + 1), 'total', step + 1)
        scheduler.step()
        # Run_loss = np.append(Run_loss, runloss / (step + 1))

    """
    test stage 
    """

    print('Start test')
    
    model.eval()
    Pn, testloss, CC, Sam, Egras = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for test_step, (test_hrMS, test_lrhs, test_ref) in enumerate(test_loader):
        test_hrMS = test_hrMS.type(torch.float).to(device)
        test_lrhs = test_lrhs.type(torch.float).to(device)
        test_ref = test_ref.type(torch.float).to(device)
        with torch.no_grad():
            test_output = model(test_lrhs, test_hrMS)
            test_loss = criteon(test_output, test_ref)
            pn = psnr(test_output.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
            # ssim += calculate_ssim(output[0].cpu().detach().numpy(), ref[0].cpu().detach().numpy())
            # if pn_test > pn:
            cc = CC_function1(test_output.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
            egras = ERGAS(test_output.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
            sam = SAM(test_output.cpu().detach().numpy(), test_ref.cpu().detach().numpy())
            # print('Sample', test_step + 1, 'loss', test_loss.item(), 'PSNR', pn, 'CC', cc, 'SAM', sam, 'EGRAS',
            #       egras)
            Pn = np.append(Pn, pn)
            CC = np.append(CC, cc)
            Sam = np.append(Sam, sam)
            Egras = np.append(Egras, egras)
            testloss = np.append(testloss, test_loss.item())
    print("epoch", i + 1, "test_loss", np.mean(testloss), 'psnr', np.mean(Pn), 'CC', np.mean(CC), 'SAM',
            np.mean(Sam), 'total', test_step + 1)
    # Test_loss = np.append(Test_loss, test_loss.cpu().detach().numpy())
    print('Finish test')
    torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict() },
                    'bestmodel.pth')


