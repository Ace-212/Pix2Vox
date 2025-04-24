import os
from torch.utils.tensorboard import SummaryWriter
from config import cfg
from core.test import test_net

cfg.CONST.WEIGHTS = 'Pix2Vox-A.pth'

cfg.DIR.OUT_PATH = './runs/%s'

os.makedirs('runs/images/test', exist_ok=True)

writer = SummaryWriter(log_dir='runs')

test_net(cfg,
         epoch_idx=0,
         output_dir=cfg.DIR.OUT_PATH,
         test_writer=writer)

writer.close()

print("Done. Check out the snapshots under runs/images/test/")