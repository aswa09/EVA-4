import numpy as np


class Config():
    def __init__(self):
        super(Config, self).__init__()
        self.seed = 1
        self.classes = None
        self.batch_size = 128
        self.nworkers = 4
        self.lr = 0.01
        self.shuffle = True
        self.momentum = 0.9
        self.epochs = 50
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        self.dataset = 'tinyimagenet'
        self.hflip_prob = 0.3
        self.vflip_prob = 0.1
        self.cutout_dim = (16, 16)
        self.cutout_prob = 0.3
        self.hue_val = 0.25
        self.rotate_lim = 10
        self.imgnt_path = None
        self.train_split = 0.7

    def forward(self):
        return self
