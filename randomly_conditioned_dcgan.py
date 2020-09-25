"""rcGAN: Learning a generative model for arbitrary size image generation.

This code borrows heavily from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

@author: Renato Barros Arantes
"""
import os
import sys
import math
import time
import argparse
import datetime
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
from sklearn.utils.random import sample_without_replacement

from PIL import Image

from abc import ABC, abstractmethod
from enum import Enum, unique

# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# pattern height
H = 64
# pattern widht
W = 64
# Number of workers for dataloader
workers = 4
# Establish convention for real and fake labels during training
REAL_LABEL = 1
FAKE_LABEL = 0
# random seed
MANUAL_SEED = 999
# debug mode
DEBUG = False
# training batch size
BATCH_SIZE = 1024
# size of z latent vector (i.e. size of generator input)
NZ = 100
# size of feature maps in generator
NGF = 64
# size of feature maps in discriminator
NDF = 64
# learning rate for optimizers
LEARNING_RATE = 0.0002
# Number of channels in the training images. RGB+Label
NC = 3

parser = argparse.ArgumentParser()
parser.add_argument('--inputImagePath', required=True,
                    help='Input image path.')
parser.add_argument('--outputFolder', required=True,
                    help='Output folder path.')
parser.add_argument('--epochs', required=False, type=int, default=500,
                    help='Number of epochs for training')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu ids: e.g. 0, use -1 for CPU')

opt = parser.parse_args()


def initialize_seeds():
    random.seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MANUAL_SEED)


def initialize_torch():
    global device, ngpu
    ngpu = torch.cuda.device_count()
    print('ngpus=%d' % (ngpu))
    print('torch.cuda.is_available=%d' % (torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('torch.version.cuda=%s' % (torch.version.cuda))
    # Decide which device we want to run on
    if opt.gpu_id != '-1':
        device = torch.device(f"cuda:{opt.gpu_id}"
                              if (torch.cuda.is_available() and ngpu > 0)
                              else "cpu")
    else:
        device = 'cpu'
    print(device, device.type)
    ######################
    cudnn.benchmark = True


def load_source_image(inputImage):
    def normalise(m):
        m = (m-m.min()) / (m.max()-m.min())
        m = m.astype(np.float32)
        return m

    print('Loading source image:', inputImage)
    m = normalise(np.array(Image.open(inputImage)))
    assert m.min() >= 0. and m.max() <= 1.
    print('I.shape={}, I.dtype={}, I.min={}, I.max={}'.format(
          m.shape, m.dtype, m.min(), m.max()))
    return m


class MyRandomSampler(Sampler):
    """Samples elements randomly without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw.
    """

    def __init__(self, data_source_len, num_samples=None):
        self.data_source_len = data_source_len
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".
                             format(self.num_samples))

    @property
    def num_samples(self):
        return self._num_samples

    def __iter__(self):
        return iter(torch.from_numpy(np.random.choice(self.data_source_len,
                                                      size=(self.num_samples,),
                                                      replace=False)))

    def __len__(self):
        return self.num_samples

class Pattern:
    def __init__(self, data):
        self.data = data.copy()

    def __eq__(self, other):
        return (self.data == other.data).all()

    def __hash__(self):
        return hash(self.data.tobytes())


class CreatePattern:
    def __init__(self, sample, pattern_height, pattern_width, rot90=False):
        self.sample = sample
        self.H = pattern_height
        self.W = pattern_width
        self.rot90 = rot90

    def __call__(self, i, j):
        b = self.sample.take(range(i, i+self.H),
                             mode='wrap', axis=0).take(range(j, j+self.W),
                                                       mode='wrap', axis=1)
        p = Pattern(b)
        if self.rot90:
            p.data = np.rot90(p.data)
        return set([p])


class PatternsFromSample:
    def __init__(self, sample, pattern_height, Pattern_width, rot90=False):
        self.sample = sample
        self.H = pattern_height
        self.W = Pattern_width
        self.rot90 = rot90
        self.number_of_patterns = 0

    def generateRandom(self, update):
        h, w = self.sample.shape[0]-self.H+1, self.sample.shape[1]-self.W+1
        cp = CreatePattern(self.sample, self.H, self.W, self.rot90)
        ix = self._randomSamples((h, w), h*w)
        for i, j in ix:
            update(cp(i, j))
            self.number_of_patterns += 1

    def _randomSamples(self, dims, n):
        idx = sample_without_replacement(np.prod(dims), n)
        return np.vstack(np.unravel_index(idx, dims)).T


def patterns_from_image_on_memory(sourceImage):
    patterns = set()
    update = lambda patts: {patterns.update(patts)}
    t0 = time.time()
    pfs = PatternsFromSample(sourceImage, H, W)
    pfs.generateRandom(update)
    patterns = list(patterns)
    random.shuffle(patterns)
    t1 = time.time()

    print('Time spent in PatternsFromSample: %.2f (s)' % (t1-t0))
    print('Number of patterns = %d' % (len(patterns)))

    return patterns

class PatternsSampleDataset(Dataset):
    def __init__(self, patterns, transform=None):
        self.transform = transform
        self.train = patterns

    def __getitem__(self, index):
        data = self.train[index].data.copy()
        data = self.transform(data)
        return (data, 1)

    def __len__(self):
        return len(self.train)


def create_sampling_dataset(patterns):
    t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = PatternsSampleDataset(patterns, t)
    n = len(dataset)
    p = int(np.ceil((n*.9)/BATCH_SIZE))*BATCH_SIZE
    print('Number of patterns = %d, but using = %d' % (n, p))
    sampler = MyRandomSampler(n, num_samples=p)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             sampler=sampler,
                                             num_workers=4)

    return dataloader


def plot_samples(dataloader):
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(
            vutils.make_grid(real_batch[0].to(device)[:32][:, 0:3, :, :],
                             nrow=8,
                             padding=2,
                             normalize=True).cpu(), (1, 2, 0)))
    plt.show()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.sz = W*H*NC
        self.createLayers()


    def createLayers(self):
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(self.sz+NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True))
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True))
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True))
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True))
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(76, NC, 4, 2, 1, bias=False),
            nn.Tanh())


    def forward(self, condition, z):
        c = condition.reshape(condition.shape[0], self.sz, 1, 1)
        i = torch.cat([c, z], 1)
        out = self.l1(i)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        c = condition.reshape(condition.shape[0], self.sz//1024, 32, 32)
        i = torch.cat([c, out], 1)
        out = self.l5(i)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF*4, NDF*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=NDF*8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.main(input)


def weights_init(m):
    """Initialize weights on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_generator():
    global netG
    # Create the generator
    netG = Generator().to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    # Print the model
    print(netG)


def create_discriminator():
    global netD
    # Create the Discriminator
    netD = Discriminator().to(device)
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    # Print the model
    print(netD)


def create_loss():
    global criterion, optimizerD, optimizerG
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE,
                            betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE,
                            betas=(0.5, 0.999))


@unique
class Conditions(Enum):
    NO_COND = 0
    HLINE_UP = 1
    HLINE_DOWN = 2
    VLINE_LEFT = 3
    VLINE_RIGHT = 4
    BOTH_HV_LINE = 5


class ConditionHandle(ABC):

    DEFAULT_MASK_VALUE = 0

    def __init__(self):
        self.sz = H*W*NC
        self.probDist = dict()
        self.probDist[Conditions.NO_COND] = 0.1
        self.probDist[Conditions.HLINE_UP] = 0.15
        self.probDist[Conditions.HLINE_DOWN] = 0.15
        self.probDist[Conditions.VLINE_LEFT] = 0.15
        self.probDist[Conditions.VLINE_RIGHT] = 0.15
        self.probDist[Conditions.BOTH_HV_LINE] = 0.30
        self.probDistValues = list(self.probDist.values())
        assert math.ceil(sum(self.probDistValues)) == 1.0, \
            print(math.ceil(sum(self.probDistValues)))

    def getCoordinates(self, size):
        d = W//5
        return np.random.randint(d, W-d, size=size)

    def getCondition(self, size):
        return np.random.choice(len(self.probDistValues), size,
                                p=self.probDistValues)

    def condHLineUp(self, I, y, condition=DEFAULT_MASK_VALUE):
        I[:, :y, :] = condition

    def condHLineDown(self, I, y, condition=DEFAULT_MASK_VALUE):
        I[:, y:, :] = condition

    def condVLineLeft(self, I, x, condition=DEFAULT_MASK_VALUE):
        I[:, :, :x] = condition

    def condVLineRight(self, I, x, condition=DEFAULT_MASK_VALUE):
        I[:, :, x:] = condition

    @abstractmethod
    def conditionBuild(self, batch):
        """Given a batch, build a condition based on it."""
        pass

    @abstractmethod
    def mergeWithGenerated(self, batch, generated):
        """Merge original and synthetic images.

        Given a generated image batch merge it with the source batch based
        on the condition
        """
        pass

class RandomConditionHandle(ConditionHandle):
    def __init__(self):
        super(RandomConditionHandle, self).__init__()
        self.GREEN = torch.tensor([0, 1, 0])
        self.GREEN_COL = self.GREEN.repeat(1, W, 1).transpose(2, 0)
        self.GREEN_ROW = self.GREEN.repeat(1, W, 1).transpose(1, 2).transpose(0, 1)
        self.condHVLineInfo = dict()

    def _condHVLine(self, I):
        y = self.getCoordinates(size=1)[0]
        x = self.getCoordinates(size=1)[0]
        quadrant = np.random.randint(4)
        if quadrant == 0:
            I[:, :y, :x] = ConditionHandle.DEFAULT_MASK_VALUE
        elif quadrant == 1:
            I[:, :y, x:] = ConditionHandle.DEFAULT_MASK_VALUE
        elif quadrant == 2:
            I[:, y:, :x] = ConditionHandle.DEFAULT_MASK_VALUE
        elif quadrant == 3:
            I[:, y:, x:] = ConditionHandle.DEFAULT_MASK_VALUE
        return [y, x, quadrant]

    def _mergeWithGenerated(self, condition, data, coordinate,
                             generated, debug):
        if condition is Conditions.NO_COND:
            data[:, :, :] = generated[:, :, :]
        elif condition is Conditions.HLINE_UP:
            self.condHLineUp(data, coordinate, generated[:, :coordinate, :])
            if debug:
                data[:, coordinate:coordinate+1, :] = self.GREEN_ROW
        elif condition is Conditions.HLINE_DOWN:
            self.condHLineDown(data, coordinate, generated[:, coordinate:, :])
            if debug:
                data[:, coordinate-1:coordinate, :] = self.GREEN_ROW
        elif condition is Conditions.VLINE_LEFT:
            self.condVLineLeft(data, coordinate, generated[:, :, :coordinate])
            if debug:
                data[:, :, coordinate-1:coordinate] = self.GREEN_COL
        elif condition is Conditions.VLINE_RIGHT:
            self.condVLineRight(data, coordinate, generated[:, :, coordinate:])
            if debug:
                data[:, :, coordinate:coordinate+1] = self.GREEN_COL
        else:
            raise Exception('Wrong condition: {}'.format(condition))

    def conditionBuild(self, batch):
        batchSize = batch.shape[0]
        # save conditions and coordinates to use latter in the replace method
        self.condHVLineInfo.clear()
        self.conditions = self.getCondition(batchSize)
        self.coordinates = self.getCoordinates(size=batchSize)
        for i in range(batchSize):
            condition = self.conditions[i]
            if Conditions(condition) is Conditions.NO_COND:
                continue
            coordinate = self.coordinates[i]
            if Conditions(condition) is Conditions.HLINE_UP:
                self.condHLineUp(batch[i], coordinate)
            elif Conditions(condition) is Conditions.HLINE_DOWN:
                self.condHLineDown(batch[i], coordinate)
            elif Conditions(condition) is Conditions.VLINE_LEFT:
                self.condVLineLeft(batch[i], coordinate)
            elif Conditions(condition) is Conditions.VLINE_RIGHT:
                self.condVLineRight(batch[i], coordinate)
            elif Conditions(condition) is Conditions.BOTH_HV_LINE:
                self.condHVLineInfo[i] = self._condHVLine(batch[i])
            else:
                raise Exception('Wrong condition: {}'.format(condition))

        return batch.reshape(batchSize, self.sz, 1, 1)

    def mergeWithGenerated(self, batch, generated, debug=False):
        batchSize = batch.shape[0]
        for i in range(batchSize):
            condition = self.conditions[i]
            if Conditions(condition) is Conditions.BOTH_HV_LINE:
                y, x, quadrant = self.condHVLineInfo[i]
                if quadrant == 0:
                    batch[i, :, :y, :x] = generated[i, :, :y, :x]
                    if debug:
                        batch[i, :, :y, x:x+1] = self.GREEN.repeat(1, y, 1).transpose(2, 0)
                        batch[i, :, y:y+1, :x] = self.GREEN.repeat(1, x, 1).transpose(1, 2).transpose(0, 1)
                elif quadrant == 1:
                    batch[i, :, :y, x:] = generated[i, :, :y, x:]
                    if debug:
                        batch[i, :, :y, x:x+1] = self.GREEN.repeat(1, y, 1).transpose(2, 0)
                        batch[i, :, y:y+1, x:] = self.GREEN.repeat(1, W-x, 1).transpose(1, 2).transpose(0, 1)
                elif quadrant == 2:
                    batch[i, :, y:, :x] = generated[i, :, y:, :x]
                    if debug:
                        batch[i, :, y:, x:x+1] = self.GREEN.repeat(1, W-y, 1).transpose(2, 0)
                        batch[i, :, y:y+1, :x] = self.GREEN.repeat(1, x, 1).transpose(1, 2).transpose(0, 1)
                elif quadrant == 3:
                    batch[i, :, y:, x:] = generated[i, :, y:, x:]
                    if debug:
                        batch[i, :, y:, x:x+1] = self.GREEN.repeat(1, W-y, 1).transpose(2, 0)
                        batch[i, :, y:y+1, x:] = self.GREEN.repeat(1, W-x, 1).transpose(1, 2).transpose(0, 1)
            else:
                coordinate = self.coordinates[i]
                self._mergeWithGenerated(Conditions(condition), batch[i],
                                         coordinate, generated[i], debug)

        return batch


##################################
conditionHandlers = [RandomConditionHandle()]


def print_time(t0=None):
    now = datetime.datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    if t0 is not None:
        print('Execution time: %.4f (h)' % ((time.time()-t0)/3600))


def get_soft_and_noisy_labels(shape, label_type, batch_size, noise_prop=0.05):
    Y = shape[2]
    X = shape[3]
    real_labels = lambda sz: np.zeros((sz, 1, Y, X)) + np.random.uniform(low=0.7, high=1.0, size=(sz, 1, Y, X))
    fake_labels = lambda sz: np.zeros((sz, 1, Y, X)) + np.random.uniform(low=0.0, high=0.3, size=(sz, 1, Y, X))
    to_flip = lambda labels: np.random.choice(np.arange(len(sn_labels)), size=int(noise_prop*len(sn_labels)))

    if label_type == REAL_LABEL:
        sn_labels = real_labels(batch_size)
        if noise_prop > 0.0:
            flipped_idx = to_flip(sn_labels)
            sn_labels[flipped_idx] = fake_labels(int(noise_prop*len(sn_labels)))
    else:
        sn_labels = fake_labels(batch_size)
        if noise_prop > 0.0:
            flipped_idx = to_flip(sn_labels)
            sn_labels[flipped_idx] = real_labels(int(noise_prop*len(sn_labels)))

    labels = torch.from_numpy(sn_labels).type(torch.FloatTensor)

    return labels.to(device)


def get_labels(shape, label_type, batch_size):
    Y = shape[2]
    X = shape[3]

    real_labels = lambda sz: np.ones((sz, 1, Y, X))
    fake_labels = lambda sz: np.zeros((sz, 1, Y, X))

    if label_type == REAL_LABEL:
        sn_labels = real_labels(batch_size)
    else:
        sn_labels = fake_labels(batch_size)

    labels = torch.from_numpy(sn_labels).type(torch.FloatTensor)

    return labels.to(device)


def save_models(epoch, iter_number, error):
    torch.save(netG.state_dict(), '{0:s}/{1:05d}_{2:05d}_gen_model.{3:.5f}.dic'.
               format(opt.outputFolder, epoch, iter_number, error))
    torch.save(netD.state_dict(), '{0:s}/{1:05d}_{2:05d}_dis_model.{3:.5f}.dic'.
               format(opt.outputFolder, epoch, iter_number, error))


def save_img_grid(img_grid, conditionHandlerName, epoch, iter_number, error):
    plt.imsave('{0:s}/img_{1:05d}_{2:05d}_{3:s}_grid.{4:.5f}.jpg'.
               format(opt.outputFolder, epoch, iter_number,
                      conditionHandlerName, error),
               img_grid)


def generate_and_save_grid(epoch, conditionHandler, fixed_grid, fixed_noise,
                           iters, errG):
    fake1 = conditionHandler.conditionBuild(fixed_grid.clone())
    netG.eval()
    with torch.no_grad():
        generated = netG(fake1, fixed_noise).detach().cpu()
    netG.train()
    # grid with green line
    fake1 = fake1.reshape(fixed_grid.shape).detach().cpu()
    grid1 = vutils.make_grid(fake1[:, 0:3, :, :], nrow=8, padding=5,
                             normalize=True)
    grid1 = np.transpose(grid1.cpu(), (1, 2, 0)).numpy()
    # normal grid
    fake2 = conditionHandler.mergeWithGenerated(fixed_grid.clone(), generated,
                                                debug=True)
    fake2 = fake2.detach().cpu()
    grid2 = vutils.make_grid(fake2[:, 0:3, :, :], nrow=8, padding=5,
                             normalize=True)
    grid2 = np.transpose(grid2.cpu(), (1, 2, 0)).numpy()
    # save the current image grid
    grid = np.hstack((grid1, grid2))
    save_img_grid(grid, conditionHandler.__class__.__name__, epoch,
                  iters, errG)


def plot_and_save_loss():
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(np.log(G_losses), label="G")
    plt.plot(np.log(D_losses), label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('{0:s}/loss.jpg'.format(opt.outputFolder))


def loadModel(model_path, model_class):
    # Create the generator
    net = model_class(ngpu).to(device)
    # Handle multi-gpu if desired
    net.load_state_dict(torch.load(model_path))
    net.train()

    return net


def loadModels(generatorPath, discriminatorPath):
    global netG, netD
    netG = loadModel(generatorPath, Generator)
    netD = loadModel(discriminatorPath, Discriminator)


def accuracy_real(out, label):
    pred_y = out >= 0.5
    num_correct = (pred_y == (label == 1)).sum().float()
    return (num_correct / len(label)).item()


def accuracy_fake(out, label):
    pred_y = out < 0.5
    num_correct = (pred_y == (label == 0)).sum().float()
    return (num_correct / len(label)).item()


def train_gan(startIters=0, startEpoch=0, numEpochs=100):
    global G_losses, D_losses, img_grid, fixed_grid, tb
    # Lists to keep track of progress
    G_losses = []
    D_losses = []

    iters = startIters
    errG = torch.tensor(0)

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)
    # Create a fixed left image to be used in visualisation
    fixed_grid = next(iter(dataloader))[0].to(device)[:64]
    generate_and_save_grid(0, conditionHandlers[0], fixed_grid,
                           fixed_noise, iters, 0)

    # Training Loop
    print_time()
    print("Starting training loop...")
    t_train = t0 = time.time()
    # For each epoch
    for epoch in range(startEpoch, numEpochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            for conditionHandler in conditionHandlers:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # **** Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                # Forward pass real batch through D
                output_real = netD(real_cpu)
                # Generate labels
                label = get_soft_and_noisy_labels(output_real.shape,
                                                  REAL_LABEL, b_size)
                # Calculate loss on all-real batch
                errD_real = criterion(output_real, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output_real.mean().item()
                # **** Train with all-fake batch
                # Generate batch of latent vectors
                z = torch.randn(b_size, NZ, 1, 1, device=device)
                # Generate fake image batch with G
                generated = netG(conditionHandler.conditionBuild(real_cpu), z)
                fake = conditionHandler.mergeWithGenerated(real_cpu, generated)
                # Classify all fake batch with D
                output_fake = netD(fake.detach())
                label = get_soft_and_noisy_labels(output_fake.shape,
                                                  FAKE_LABEL, b_size)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output_fake, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = (errD_real + errD_fake) / 2
                # Update D
                optimizerD.step()
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                # Since we just updated D, perform another forward pass
                # of all-fake batch through D
                output_fake = netD(fake)
                # fake labels are real for generator cost
                label = get_soft_and_noisy_labels(output_real.shape,
                                                  REAL_LABEL, b_size)
                # Calculate G's loss based on this output
                errG = criterion(output_fake, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output_fake.mean().item()
                # Update G
                optimizerG.step()

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f - time %.4f'
                  % (epoch+1, numEpochs, i+1, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                     time.time()-t0))
            t0 = time.time()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        for conditionHandler in conditionHandlers:
            generate_and_save_grid(epoch, conditionHandler, fixed_grid,
                                   fixed_noise, iters, G_losses[-1])

        if epoch > 0 and epoch % 50 == 0:
            save_models(epoch, iters, G_losses[-1])

    # Save the final models and samples
    save_models(epoch, iters, G_losses[-1])
    for conditionHandler in conditionHandlers:
        generate_and_save_grid(epoch, conditionHandler, fixed_grid,
                               fixed_noise, iters, G_losses[-1])

    print_time(t_train)


def getModelsPath():
    global startIters, startEpoch
    startIters, startEpoch = -1, -1
    genName, disName = None, None
    for root, directories, files in os.walk(opt.outputFolder):
        for f in sorted(files):
            if f.endswith('.dic'):
                l = f.split('_')
                epoch = int(l[0])
                itern = int(l[1])
                if epoch >= startEpoch and itern >= startIters:
                    genName = '%s_%s_gen_%s' % (l[0], l[1], l[3])
                    disName = '%s_%s_dis_%s' % (l[0], l[1], l[3])
                    startEpoch = epoch
                    startIters = itern
    startIters += 1
    startEpoch += 1
    if genName is not None and disName is not None:
        return os.path.join(opt.outputFolder, genName), \
               os.path.join(opt.outputFolder, disName)
    else:
        return None, None


def createModels():
    print('Creating generator...')
    create_generator()
    print('Creating discriminator...')
    create_discriminator()


def createDataset(inputImagePath):
    sourceImage = load_source_image(inputImagePath)
    patterns = patterns_from_image_on_memory(sourceImage)
    return create_sampling_dataset(patterns)


def initialise(inputImagePath):
    global sourceImage, patterns, dataloader, startIters, startEpoch
    initialize_seeds()
    initialize_torch()
    startIters, startEpoch = 0, 0
    if os.path.exists(opt.outputFolder):
        generatorPath, discriminatorPath = getModelsPath()
        print(generatorPath, discriminatorPath)
        if generatorPath is not None and discriminatorPath is not None:
            loadModels(generatorPath=generatorPath,
                       discriminatorPath=discriminatorPath)
            create_loss()
            if startIters > 0 and startEpoch > 0:
                if startEpoch >= opt.epochs:
                    print('(%d/%d) Nothing to do here!:)' %
                          (startEpoch, opt.epochs))
                    sys.exit(0)
                print('Continuing training from epoch = %d and iteration = %d.'
                      % (startEpoch, startIters))
                dataloader = createDataset(inputImagePath)
                return startIters, startEpoch
    print('Starting training from scratch!')
    os.makedirs(opt.outputFolder, exist_ok=True)
    dataloader = createDataset(inputImagePath)
    createModels()
    create_loss()
    return startIters, startEpoch


if __name__ == '__main__':
    startIters, startEpoch = initialise(opt.inputImagePath)
    train_gan(startIters=startIters, startEpoch=startEpoch,
              numEpochs=opt.epochs)
    plot_and_save_loss()
