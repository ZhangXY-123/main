import argparse
from utils.trainer import trainer
from utils.dataset import LITS_dataset, make_dataloaders
import torch
import torch.nn as nn
# from model.unet import UNet
import numpy as np
import torch.nn as nn
# from Dice import DiceLoss
# from loss.WBCE import WCELoss
#from loss.Tversky import TverskyLoss
from EfficientNet import EfficientNet
from EfficientUnet_git_B4_res import Unet
from pq.UNET import UNet
from TVERSKY import  FocalTversky_loss, TverskyLoss
# from LOSS1 import DiceLosss
from Net.FCN import VGGNet, FCNs
# from BLDice import DiceLoss
from Tversky1 import TverskyLoss
from Net.AttentionUnet import AttU_Net
# from Net.DenseUnet import DenseUNet
from Net.EU import Unet
from pp.Dice import DiceLoss
# from Net.asd import Se_PPP_ResUNet
# from Net.Res_Att_Unet import AttU_Net
# from Net.DenseUnet import DenseUNet
# from Net.PP import SE_P_AttU_Net
from Net.Unenpplus import NestedUNet
# from Net.SegNet import SegNet
from Net.ERA_Unet import Unet
from Net.Wbceloss import WCELoss
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LEARNING_RATE = 1e-3
    LR_DECAY_STEP = 2
    LR_DECAY_FACTOR = 0.5
    WEIGHT_DECAY = 5e-4
    BATCH_SIZE = 4
    MAX_EPOCHS = 60
    # MODEL = SegNet(1, 2).to(device)
    # MODEL = NestedUNet(1, 2).to(device)
    # MODEL = DenseUNet(2).to(device)
    # MODEL = DeepResUNet(2).to(device)
    # MODEL = AttU_Net(1, 2).to(device)
    # MODEL = SeResUNet(1, 2, deep_supervision=False).to(device)
    # MODEL = SE_P_AttU_Net(1, 2).to(device)
    # MODEL = UNet(1, 2).to(device)
    # vgg_model = VGGNet()
    # MODEL = FCNs(pretrained_net=vgg_model, n_class=2).to(device)
    # MODEL = ResUnetPlusPlus(1,filters=[32, 64, 128, 256, 512]).to(device)
    # MODEL = Res_UNet(1,2).to(device)
    efficient_model = EfficientNet()
    MODEL = Unet(out_ch=2, pretrained_net=efficient_model)
    # MODEL = Se_PPP_ResUNet(1, 2, deep_supervision=False).to(device)
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    LR_SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    # CRITERION = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.75,1])).float(),size_average=True).to(device)
    #CRITERION = nn.CrossEntropyLoss().to(device)
    CRITERION = WCELoss().to(device)
    # CRITERION = TverskyLoss().to(device)
    # CRITERION = WCELoss().to(device)

    tr_path_raw = 'D:/bpq/tr/raw'
    tr_path_label = 'D:/bpq/tr/label'
    ts_path_raw = 'D:/bpq/ts/raw'
    ts_path_label = 'D:/bpq/ts/label'

    # checkpoints_dir = 'checkpoints'
    checkpoints_dir = 'final checkpoints/B4'
    checkpoint_frequency = 1000
    dataloaders = make_dataloaders(tr_path_raw, tr_path_label, ts_path_raw, ts_path_label, BATCH_SIZE, n_workers=0)
    comment = 'liver_segmentation_B4'
    verbose_train = 1
    verbose_val = 500

    trainer = trainer(MODEL, OPTIMIZER, LR_SCHEDULER, CRITERION, dataloaders, comment, verbose_train, verbose_val, checkpoint_frequency, MAX_EPOCHS, checkpoint_dir=checkpoints_dir, device=device)
    trainer.train()
