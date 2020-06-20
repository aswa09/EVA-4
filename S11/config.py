class Config():
    def __init__(self):
        super(Config, self).__init__()
        self.seed = 1
        self.classes=['plane', 'car', 'bird', 'cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck']
        self.batch_size= 512
        self.max_at_epoch=5
        self.nworkers = 4
        self.start_lr = 1e-6
        self.end_lr = 0.02
        self.shuffle = True
        self.momentum = 0.9
        self.weight_decay = 0.01
        self.epochs = 24

        def forward(self):
            return self