import argparse
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from torch.nn import functional as F

# import visdom
import torch.optim as optim
from data.load_data import load_data

from torchvision.utils import make_grid

from utils.utils import combine_images, save_image
from tensorboardX import SummaryWriter

# vis = visdom.Visdom()
# vis.env = 'vae_dcgan'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='cifar10 | lsun | imagenet | folder | lfw ')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=50, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='model', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
#
# cudnn.benchmark = True
#
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataloader = torch.utils.data.DataLoader(load_data(opt.dataset, img_size=opt.imageSize), batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = torch.exp(0.5 * logvar)
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, img_size):
        super(_Encoder, self).__init__()

        n = math.log2(img_size)
        #
        assert n==round(n),'img_size must be a power of 2'
        # assert n>=3,'img_size must be at least 8'
        n=int(n)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv',
                                nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu',
                                nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** i, ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        # state size. (ngf*8) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        return [self.conv1(output), self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()

        ngf = 7

        n = int(math.log2(imageSize))

        # assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        # n = int(n)

        self.decoder = nn.Sequential()
        # self.decoder.add_module("resize", nn.Res)
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 3)))
        self.decoder.add_module('input-relu', nn.ReLU(inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i - 1)), nn.ReLU(inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        # self.decoder.add_module('Sigmoid', nn.Sigmoid())
        self.decoder.add_module('output-tanh', nn.Tanh())

    def forward(self, input):
        output = self.encoder(input)
        output = self.sampler(output)
        output = self.decoder(output)
        return output

    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n - 3):
            self.main.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** (i), ngf * 2 ** (i + 1)),
                                 nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)), nn.BatchNorm2d(ndf * 2 ** (i + 1)))
            self.main.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

netG = _netG(opt.imageSize, ngpu)
netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))

netD = _netD(opt.imageSize,ngpu)
netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
#
criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()


if opt.cuda:
    print("OK")
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    input, label = input.cuda(), label.cuda()
    print("OK")
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
#
# # setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#
writer = SummaryWriter()

#
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        optimizerD.zero_grad()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(real_cpu.size(0), 1).fill_(real_label)

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

            # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        gen = netG.decoder(noise)

        # if i == 0:
        #     save_image(combine_images(gen), f"images/{epoch}_{i}.png")
        # gen_win = vis.image(gen.data[0].cpu()*0.5+0.5,win = gen_win)
        label.data.fill_(fake_label)
        output = netD(gen.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: VAE
        ###########################
        optimizerG.zero_grad()
        encoded = netG.encoder(input)
        mu = encoded[0]
        logvar = encoded[1]

        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        KLD /= (opt.batchSize * 32 * 32)

        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled)
        # if i == 0:
        #     save_image(combine_images(rec), f"reconstructions/{epoch}_{i}.png")
        # rec_win = vis.image(rec.data[0].cpu()*0.5+0.5,win = rec_win)
        # zero_to_one_input = (input + 1.) / 2.

        # rec_err = F.binary_cross_entropy(rec, input, size_average=False)
        rec_err = F.mse_loss(rec, input, size_average=True)

        # VAEerr = KLD + MSEerr
        VAEerr = (rec_err + KLD)

        scaled_VAEerr = 5. * VAEerr

        scaled_VAEerr.backward()
        optimizerG.step()

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        #
        rec = netG(input)  # this tensor is freed from mem at this point
        output = netD(rec)
        errG = criterion(output, label)

        noise.data.normal_(0, 1)
        new_gen = netG.decoder(noise)
        errG += criterion(netD(new_gen), label)

        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()


        if i % 10 == 0:
            print('[%d/%d][%d/%d] Rec loss: %.4f Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader), rec_err.item(),
                     VAEerr.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            step = epoch * len(dataloader) + i
            writer.add_scalar("losses/Reconstruction", rec_err.item(), step)
            writer.add_scalar("losses/VAE", VAEerr.item(), step)
            writer.add_scalar("losses/Generator", errG.item(), step)
            writer.add_scalar("losses/Discriminator", errD.item(), step)
            writer.add_scalar("Probability/D_x", D_x, step)
            writer.add_scalar("Probability/D_G_z", D_G_z1, step)
            writer.add_scalar("Probability/D_G_E_z", D_G_z2, step)
    #
    writer.add_image("Images/samples", make_grid((gen + 1) / 2, nrow=16), global_step=epoch)
    writer.add_image("Images/original", make_grid((input + 1) / 2, nrow=16), global_step=epoch)
    writer.add_image("Images/reconstructions", make_grid((rec + 1) / 2, nrow=16), global_step=epoch)

    # if epoch%opt.saveInt == 0 and epoch!=0:
    #     torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #     torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
