import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torchvision.datasets as dset


def load_data(dataset_name, data_root=None, img_size=None):

    if dataset_name in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        return dset.ImageFolder(root=data_root,
                                   transform=transforms.Compose([
                                       transforms.Resize(img_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    if dataset_name == 'mnist':
        return MNIST(root='data/mnist',
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
        # elif opt.dataset == 'lsun':
        #     dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
        #                         transform=transforms.Compose([
        #                             transforms.Scale(opt.imageSize),
        #                             transforms.CenterCrop(opt.imageSize),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                         ]))
        # elif opt.dataset == 'cifar10':
        #     dataset = dset.CIFAR10(root=opt.dataroot, download=True,
        #                            transform=transforms.Compose([
        #                                transforms.Scale(opt.imageSize),
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #                            ])
    #     )
