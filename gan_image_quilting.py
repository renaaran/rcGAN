"""
To generate images with a pre-trained rcGAN model.

This is a mix of the image quilting algorithm over GAN generated patches.

@author: Renato B. Arantes
"""
import os
import sys
import heapq
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn

from concurrent import futures
from itertools import product

# size of z latent vector (i.e. size of generator input)
NZ = 100
BLOCK_SIZE = 64
OVERLAP_SIZE = int(BLOCK_SIZE/2)
# size of feature maps in generator
NGF = 64
# size of feature maps in discriminator
NDF = 64
# Number of channels in the training images. RGB+Label
NC = 3

parser = argparse.ArgumentParser()
parser.add_argument('--inputImagePath', required=True,
                    help='Input image path.')
parser.add_argument('--outputFolder', required=True,
                    help='Output folder path.')
parser.add_argument('--modelPath', required=True,
                    help='rcGAN generator model path.')
parser.add_argument('--numberOfTiles', required=True, type=int, nargs='+',
                    default=[5, 5],
                    help=('output image width in tiles of 64x64,'
                          'change it to change generated image size.'))
parser.add_argument('--n', required=True, type=int,
                    default=5, help='Number of images to generate.')

opt = parser.parse_args()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.sz = 64*64*3
        self.createLayers()

    def createLayers(self):
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(self.sz+NZ, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True))
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True))
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True))
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
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


class PatternsDataset():
    def __init__(self, patterns, transform=None):
        self.patterns = patterns
        self.count = len(self.patterns)
        self.transform = transform

    def __getitem__(self, index):
        data = self.patterns[index].data.astype(np.float32)
        data = self.transform(data)
        return (data, 0)

    def __len__(self):
        return self.count


class Pattern:
    def __init__(self, data):
        self.data = data.copy().astype(np.float32)

    def __eq__(self, other):
        return (self.data == other.data).all()

    def __hash__(self):
        return hash(self.data.tobytes())


class CreatePattern:
    def __init__(self, sample, N, ref=False, rot=False):
        self.sample = sample
        self.ref = ref
        self.rot = rot
        self.N = N

    def __call__(self, t):
        i = t[0]
        j = t[1]
        t = Pattern(self.sample.
                    take(range(i, i+self.N), mode='raise', axis=0).
                    take(range(j, j+self.N), mode='raise', axis=1))
        res = set([t])
        if self.ref:
            res.add(Pattern(np.fliplr(t.data)))
        if self.rot:
            res.add(Pattern(np.rot90(t.data)))
        return os.getpid(), res


# define the possible tiles orientations
class Orientation:
    RIGHT_LEFT = 1
    BOTTOM_TOP = 2
    BOTH = 3


class Minimum_Cost_Path:
    def __init__(self, blk1, blk2, overlap_size, orientation):
        assert blk1.shape == blk2.shape
        assert blk1.shape[0] == blk2.shape[1]
        # get the overlap regions
        block_size = blk1.shape[0]
        # calculate LE error for the overlap region
        self.L2_error = self.calc_L2_error(blk1, blk2, block_size,
                                           overlap_size, orientation)
        # calculate the minimum cost matrix
        self.cost = np.zeros(self.L2_error.shape, dtype=np.float32)
        self.calc_cost()
        # now calculate the minimum cost path
        self.path = self.minimum_cost_path()

    def get_cost_at(self, i, j):
        if i < 0 or i >= self.cost.shape[0] or \
            j <= 0 or \
                j >= self.cost.shape[1]-1:
            return sys.maxsize
        return self.cost[i][j]

    def get_costs(self, i, j):
        x = self.get_cost_at(i-1, j-1)
        y = self.cost[i-1][j]
        z = self.get_cost_at(i-1, j+1)
        return x, y, z

    def min_index(self, i, j):
        x, y, z = self.get_costs(i, j)
        if (x < y):
            return j-1 if (x < z) else j+1
        else:
            return j if (y < z) else j+1

    def minimum_cost_path(self):
        rows, _ = self.cost.shape
        p = [np.argmin(self.cost[rows-1, :])]
        for i in range(rows-1, 0, -1):
            j = p[-1]
            # get the index of smaller cost
            p.append(self.min_index(i, j))
        p.reverse()
        return p

    def calc_cost(self):
        # we don't need to calculate the first row
        self.cost[0, :] = self.L2_error[0, :]
        rows, cols = self.cost.shape
        for i in range(1, rows):
            for j in range(cols):
                x, y, z = self.get_costs(i, j)
                self.cost[i][j] = min(x, y, z) + self.L2_error[i][j]

    @staticmethod
    def get_overlap(blk1, blk2, block_size, overlap_size, orientation):
        if orientation == Orientation.RIGHT_LEFT:
            ov1 = blk1[:, -overlap_size:, :3]  # right
            ov2 = blk2[:, :overlap_size, : 3]  # left
        elif orientation == Orientation.BOTTOM_TOP:
            # bottom
            ov1 = np.transpose(blk1[-overlap_size:, :, : 3], (1, 0, 2))
            # top
            ov2 = np.transpose(blk2[:overlap_size, :, : 3], (1, 0, 2))
        assert ov1.shape == ov2.shape
        return ov1, ov2

    @staticmethod
    def calc_L2_error(blk1, blk2, block_size, overlap_size, orientation):
        ov1, ov2 = Minimum_Cost_Path.get_overlap(blk1, blk2, block_size,
                                                 overlap_size, orientation)
        L2_error = np.sum((ov1-ov2)**2, axis=2)
        assert (L2_error >= 0).all() == True
        return L2_error


class Image_Quilting:
    def __init__(self, source_image, generator_model, block_size,
                 overlap_size, number_of_tiles_in_output):
        self.source_image = self.normalizeBetweenMinus1and1(source_image)
        self.generator_model = generator_model
        self.block_size = block_size
        self.overlap_size = overlap_size
        self.number_of_tiles_in_output = number_of_tiles_in_output

        self.image_size = [0, 0]
        self.image_size[0] = (2*(block_size-overlap_size)) + \
            ((number_of_tiles_in_output[0]-2)*(block_size-2*overlap_size)) + \
            ((number_of_tiles_in_output[0]-1)*overlap_size)
        self.image_size[1] = (2*(block_size-overlap_size)) + \
            ((number_of_tiles_in_output[1]-2)*(block_size-2*overlap_size)) + \
            ((number_of_tiles_in_output[1]-1)*overlap_size)
        self.image_channels = source_image.shape[2]

        self.patterns = self.patterns_from_sample(self.source_image)
        np.random.shuffle(self.patterns)

    def __save_debug_image(self, title, img, image_name=None):
        img = self.normalizeBetween0and1(img)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(title)
        plt.imshow(img)
        plt.show()
        if image_name is not None:
            plt.imsave(image_name, img)

    @staticmethod
    def normalizeBetweenMinus1and1(_m):
        m = _m.copy().astype(np.float32)
        m = -1+2*(m-m.min())/(m.max()-m.min())
        assert m.min() >= -1. and m.min() < 0. and m.max() > 0. and m.max() <= 1.
        return m

    @staticmethod
    def normalizeBetween0and1(_m):
        m = _m.copy().astype(np.float32)
        m = (m-m.min()) / (m.max()-m.min())
        assert m.min() >= 0. and m.min() <= 1.
        return m

    def patterns_from_sample(self, source_image):
        patts = set()
        N = self.block_size
        h, w, _ = source_image.shape
        with futures.ProcessPoolExecutor() as pool:
            createPattern = CreatePattern(source_image, N)
            for _, toappend in pool.map(createPattern,
                                        product(range(0, h-N), range(0, w-N)),
                                        chunksize=w):
                patts.update(toappend)
        return list(patts)

    def join_horizontal_blocks(self, blk1, blk2, path, debug=False):
        G = [[0, -1., -1.]]  # red pixel
        sl1 = blk1[:, -self.overlap_size:]
        sl2 = blk2[:, :self.overlap_size]
        a = path[0]
        if debug:
            join_row = lambda i: np.concatenate((sl1[i, :max(0, a-1)], G,
                                                 sl2[i, max(a, 1):]))
        else:
            join_row = lambda i: np.concatenate((sl1[i, :a], sl2[i, a:]))
        c = join_row(0)
        res = np.zeros((self.block_size, self.overlap_size,
                        self.image_channels), dtype=np.float32)
        res[0, :] = c
        for i in range(1, self.block_size):
            a = path[i]
            c = join_row(i)
            res[i, :] = c
        if debug:
            self.__save_debug_image('Join Horizontal',
                                    np.hstack((res,
                                               blk2[:, self.overlap_size:])))
        return np.hstack((res, blk2[:, self.overlap_size:]))

    def join_vertical_blocks(self, blk1, blk2, path, debug=False):
        G = [[0, -1., -1.]]  # red pixel
        sl1 = blk1[-self.overlap_size:, :]
        sl2 = blk2[:self.overlap_size, :]
        a = path[0]
        if debug:
            join_col = lambda i: np.concatenate((sl1[: max(0, a-1),i], G,
                                                 sl2[max(a, 1):, i]))
        else:
            join_col = lambda i: np.concatenate((sl1[:a, i], sl2[a:, i]))
        c = join_col(0)
        res = np.zeros((self.overlap_size, self.block_size,
                        self.image_channels), dtype=np.float32)
        res[:, 0] = c
        for i in range(1, self.block_size):
            a = path[i]
            c = join_col(i)
            res[:, i] = c
        if debug:
            self.__save_debug_image('Join Vertical',
                                    np.vstack((res, blk2[self.overlap_size:, :])))
        return np.vstack((res, blk2[self.overlap_size:, :]))

    def get_random_pattern(self):
        y, x = np.random.randint(0, 50, 2)
        blk = self.source_image[y:y+self.block_size, x:x+self.block_size, :]
        return blk

    def old_get_random_pattern(self):
        i = np.random.randint(0, high=len(self.patterns))
        return self.patterns[i].data

    def get_gen_block(self, condition, z):
        x = self.generator_model(condition, z)
        y = np.array(x.to("cpu").permute([0, 2, 3, 1]).detach().numpy())
        return y

    def get_GAN_genetared_samples(self, condition, N):
        z = torch.randn(N, NZ, 1, 1, device=device)
        blk = torch.from_numpy(np.transpose(np.array([condition]),
                                            (0, 3, 1, 2))).to(device)
        b = blk.repeat(N, 1, 1, 1)
        if self.debug:
            self.condition = condition.copy()
            self.__save_debug_image('Condition', self.condition)
        return [Pattern(b) for b in self.get_gen_block(b, z)]

    def get_samples(self, blks, orientation):
        # 1% probability of selecting a real patch.
        if np.random.rand() < .99:
            N = 100
            if orientation != Orientation.BOTH:
                blk = np.zeros((self.block_size, self.block_size,
                                self.image_channels), dtype=np.float32)
                if orientation == Orientation.BOTTOM_TOP:
                    blk[:self.overlap_size, :] = blks[0][-self.overlap_size:, :]
                else:
                    blk[:, :self.overlap_size] = blks[0][:, -self.overlap_size:]
            else:
                tmp = np.zeros((self.block_size*2, self.block_size*2,
                                self.image_channels), dtype=np.float32)
                tmp[:self.block_size, :self.block_size] = blks[2]
                tmp[self.overlap_size:self.block_size+self.overlap_size,
                    :self.block_size] = blks[1]
                tmp[:self.block_size,
                    self.overlap_size:self.block_size+self.overlap_size] = blks[0]
                blk = tmp[self.overlap_size:self.block_size+self.overlap_size,
                          self.overlap_size:self.block_size+self.overlap_size]

            return self.get_GAN_genetared_samples(blk, N)
        else:
            return np.random.choice(self.patterns,
                                    size=self.sample_size, replace=False)

    def get_best(self, blks, orientation):
        pq = []
        pq_N = 1
        heapq.heapify(pq)
        samples = self.get_samples(blks, orientation)
        for patt in samples:
            blk = patt.data
            if orientation != Orientation.BOTH:
                l2 = Minimum_Cost_Path.calc_L2_error(blks[0], blk,
                                                     self.block_size,
                                                     self.overlap_size,
                                                     orientation)
                err = l2.sum()
            else:
                l2u = Minimum_Cost_Path.calc_L2_error(blks[0], blk,
                                                      self.block_size,
                                                      self.overlap_size,
                                                      Orientation.BOTTOM_TOP)
                l2l = Minimum_Cost_Path.calc_L2_error(blks[1], blk,
                                                      self.block_size,
                                                      self.overlap_size,
                                                      Orientation.RIGHT_LEFT)
                err = l2u.sum() + l2l.sum()
            pqe = (-err, blk)
            if len(pq) < pq_N:
                heapq.heappush(pq, pqe)
            else:
                try:
                    heapq.heappushpop(pq, pqe)
                except ValueError:
                    # skip errors related to duplicate values
                    None
        idx = np.random.choice(len(pq), 1)[0]
        if self.debug:
            self.best = pq[idx][1].copy()
            self.__save_debug_image('Best', self.best)
        return pq[idx][1]

    def add_block(self, blk, y, x):
        dx = max(0, x*(self.block_size-self.overlap_size))
        dy = max(0, y*(self.block_size-self.overlap_size))
        self.output_image[dy:dy+self.block_size,
                          dx:dx+self.block_size, :] = blk.copy()

    def get_block(self, y, x):
        dx = max(0, x*(self.block_size-self.overlap_size))
        dy = max(0, y*(self.block_size-self.overlap_size))
        return self.output_image[dy:dy+self.block_size,
                                 dx:dx+self.block_size, :].copy()

    def generate(self, sample_size=1, debug=False, show_progress=False):
        self.debug = debug
        self.sample_size = int(np.ceil(len(self.patterns)*sample_size))
        self.output_image = np.zeros((self.image_size[0], self.image_size[1],
                                      self.image_channels), dtype=np.float32)
        for i in range(self.number_of_tiles_in_output[0]):
            self.row = i
            for j in range(self.number_of_tiles_in_output[1]):
                self.col = j
                if show_progress:
                    print('\rProgress : (%d,%d)  ' % (i+1, j+1),
                          end='', flush=True)
                if i == 0 and j == 0:
                    self.output_image[:self.block_size, :self.block_size] = \
                        self.get_random_pattern()
                elif i == 0 and j > 0:
                    blk1 = self.get_block(0, j-1)  # up
                    blk2 = self.get_best((blk1,), Orientation.RIGHT_LEFT)
                    mcp = Minimum_Cost_Path(blk1, blk2, self.overlap_size,
                                            Orientation.RIGHT_LEFT)
                    out = self.join_horizontal_blocks(blk1, blk2, mcp.path)
                    self.add_block(out, i, j)
                    ##################
                    if self.debug:
                        out = self.join_horizontal_blocks(blk1, blk2, mcp.path,
                                                          debug=True)
                        pad = np.ones((blk1.shape[0], 2, 3))
                        img = np.hstack((self.condition, pad,
                                         blk2[:, :, :3], pad,
                                         out))
                        self.__save_debug_image('Column', img,
                                                os.path.join(opt.outputFolder,
                                                'join_{}_col_{}.png').
                                                format(i, j))
                elif i > 0 and j == 0:
                    blk1 = self.get_block(i-1, 0)  # left
                    blk2 = self.get_best((blk1,), Orientation.BOTTOM_TOP)
                    mcp = Minimum_Cost_Path(blk1, blk2, self.overlap_size,
                                            Orientation.BOTTOM_TOP)
                    out = self.join_vertical_blocks(blk1, blk2, mcp.path)
                    self.add_block(out, i, j)
                    ##################
                    if self.debug:
                        out = self.join_vertical_blocks(blk1, blk2, mcp.path,
                                                        debug=True)
                        pad = np.ones((blk1.shape[0], 2, 3))
                        img = np.hstack((self.condition, pad,
                                         blk2[:, :, :3], pad,
                                         out))
                        self.__save_debug_image('Row', img,
                                                os.path.join(opt.outputFolder,
                                                'join_{}_col_{}.png').
                                                format(i, j))
                elif i > 0 and j > 0:
                    blk1 = self.get_block(i-1, j)  # up
                    blk2 = self.get_block(i, j-1)  # left
                    blk3 = self.get_block(i-1, j-1)  # corner
                    blk4 = self.get_best((blk1, blk2, blk3), Orientation.BOTH)
                    mcp1 = Minimum_Cost_Path(blk1, blk4, self.overlap_size,
                                             Orientation.BOTTOM_TOP)
                    mcp2 = Minimum_Cost_Path(blk2, blk4, self.overlap_size,
                                             Orientation.RIGHT_LEFT)
                    assert mcp1 != mcp2
                    out1 = self.join_vertical_blocks(blk1, blk4, mcp1.path)
                    out2 = self.join_horizontal_blocks(blk2, out1, mcp2.path)
                    out1.shape == out2.shape
                    self.add_block(out2, i, j)
                    ##################
                    if self.debug:
                        out1 = self.join_vertical_blocks(blk1, blk4,
                                                         mcp1.path, debug=True)
                        out2 = self.join_horizontal_blocks(blk2, out1,
                                                         mcp2.path, debug=True)
                        pad = np.ones((blk1.shape[0], 2, 3))
                        img = np.hstack((self.condition, pad,
                                         blk4, pad,
                                         out2, pad))
                        self.__save_debug_image('Corner', img,
                                                os.path.join(opt.outputFolder,
                                                'join_{}_col_{}.png').
                                                format(i, j))

                if self.debug:
                    self.__save_debug_image('Result',
                                            self.output_image[:, :, :3],
                                            os.path.join(opt.outputFolder,
                                            'row_{}_col_{}.png').
                                            format(i, j))

        return self.normalizeBetween0and1(self.output_image)


def load_source_image(source_path):
    img = plt.imread(source_path)
    if np.max(img) > 1.:
        img = Image_Quilting.normalizeBetween0and1(img)

    print('img.shape={}, img.dtype={}, img.max={}, img.min={}'.format(
          img.shape, img.dtype, img.max(), img.min()))

    assert img.min() >= 0. and img.max() <= 1.

    return img


def initialise_torch():
    global device, ngpu
    ngpu = torch.cuda.device_count()
    print('ngpus=%d' % (ngpu))
    print('torch.cuda.is_available=%d' % (torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('torch.version.cuda=%s' % (torch.version.cuda))
    # Decide which device we want to run on
    device = torch.device("cuda:0"
                          if (torch.cuda.is_available() and ngpu > 0)
                          else "cpu")
    print(device)
    ######################
    cudnn.benchmark = True


def load_model(model_path, model_class):
    # Create the generator
    netG = model_class().to(device)

    # Handle multi-gpu if desired
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    return netG


def load_generator_models():
    return load_model(opt.modelPath, model_class=Generator)


def initialise():
    global netG, source_image, uniqueLabels, linearSpace
    initialise_torch()
    netG = load_generator_models()
    source_image = load_source_image(opt.inputImagePath)


if __name__ == '__main__':
    initialise()
    iq = Image_Quilting(source_image, netG, BLOCK_SIZE, OVERLAP_SIZE,
                        (opt.numberOfTiles[0], opt.numberOfTiles[1]))
    print('Number of patterns = {}'.format(len(iq.patterns)))
    #
    for i in range(opt.n):
        output_image = iq.generate(show_progress=True)
        plt.axis("off")
        plt.imsave(os.path.join(opt.outputFolder, '{}_iq_{}x{}.png').
                   format(i, opt.numberOfTiles[0], opt.numberOfTiles[1]),
                   output_image)
        print()

        plt.close()
