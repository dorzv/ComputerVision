import sys
import os
import datetime
import time
from random import choice
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import torchvision.utils
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

plt.style.use('dark_background')
px = 1 / plt.rcParams['figure.dpi']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.5,
}

def get_training_data(images, batch_size):
    """
    Create data for training. Make a warp and target version of the images.
    Args:
        images (tuple[np.array]): a tuple with images as numpy array
        batch_size (int): the number of wrap images and target images to return

    Returns:
        (np.array): a matrix of size (batch_size, 96, 96, 3) of warp images
        (np.array): a matrix of size (batch_size, 96, 96, 3) of target images
    """
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image)

        # first time initialize empty array
        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    """
    Make a random transformation for an image, including rotation, zoom, shift and flip.
    Args:
        image (np.array): an image to be transformed
        rotation_range (float): the range of possible angles to rotate - [-rotation_range, rotation_range]
        zoom_range (float): range of possible scales - [1 - zoom_range, 1 + zoom_range]
        shift_range (float): the percent of translation for x  and y
        random_flip (float): the probability of horizontal flip

    Returns:
        (np.array): transformed image
    """
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

def random_warp(image):
    """
    Create a distorted face image and a target undistorted image
    Args:
        image  (np.array): image to warp

    Returns:
        (np.array): warped version of the image
        (np.array): target image to construct from the warped version
    """
    h, w = image.shape[:2]

    # build coordinate map to wrap the image according to
    range_ = np.linspace(h / 2 - h * 0.4, h / 2 + h * 0.4, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    # add noise to get a distortion of the face while warp the image
    mapx = mapx + np.random.normal(size=(5, 5), scale=5*h/256)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5*h/256)

    # get interpolation map for the center of the face with size of (96, 96)
    interp_mapx = cv2.resize(mapx, (int(w / 2 * (1 + 0.25)) , int(h / 2 * (1 + 0.25))))[int(w/2 * 0.25/2):int(w / 2 * (1 + 0.25) - w/2 * 0.25/2), int(w/2 * 0.25/2):int(w / 2 * (1 + 0.25) - w/2 * 0.25/2)].astype('float32')
    interp_mapy = cv2.resize(mapy, (int(w / 2 * (1 + 0.25)) , int(h / 2 * (1 + 0.25))))[int(w/2 * 0.25/2):int(w / 2 * (1 + 0.25) - w/2 * 0.25/2), int(w/2 * 0.25/2):int(w / 2 * (1 + 0.25) - w/2 * 0.25/2)].astype('float32')

    # remap the face image according to the interpolation map to get warp version
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    # create the target (undistorted) image
    # find a transformation to go from the source coordinates to the destination coordinate
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:w//2+1:w//8, 0:h//2+1:h//8].T.reshape(-1, 2)

    # We want to find a similarity matrix (scale rotation and translation) between the
    # source and destination points. The matrix should have the structure
    # [[a, -b, c],
    #  [b,  a, d]]
    # so we can construct unknown vector [a, b, c, d] and solve for it using least
    # squares with the source and destination x and y points.
    A = np.zeros((2 * src_points.shape[0], 2))
    A[0::2, :] = src_points  # [x, y]
    A[0::2, 1] = -A[0::2, 1] # [x, -y]
    A[1::2, :] = src_points[:, ::-1]  # [y, x]
    A = np.hstack((A, np.tile(np.eye(2), (src_points.shape[0], 1))))  # [x, -y, 1, 0] for x coordinate and [y, x, 0 ,1] for y coordinate
    b = dst_points.flatten()  # arrange as [x0, y0, x1, y1, ..., xN, yN]

    similarity_mat = np.linalg.lstsq(A, b, rcond=None)[0] # get the similarity matrix elements as vector [a, b, c, d]
    # construct the similarity matrix from the result vector of the least squares
    similarity_mat = np.array([[similarity_mat[0], -similarity_mat[1], similarity_mat[2]],
                               [similarity_mat[1], similarity_mat[0], similarity_mat[3]]])
    # use the similarity matrix to construct the target image using affine transformation
    target_image = cv2.warpAffine(image, similarity_mat, (w // 2, h // 2))

    return warped_image, target_image


class FaceData(Dataset):
    def __init__(self, data_path):
        self.image_files_src = glob(data_path + '/src/aligned/*.jpg')
        self.image_files_dst = glob(data_path + '/dst/aligned/*.jpg')

    def __len__(self):
        return min(len(self.image_files_src), len(self.image_files_dst))

    def __getitem__(self, inx):
        """
        Choose randomly source and destination images, resize them to
        2*96x2*96, and convert to numpy array with values between 0 and 1
        Args:
            inx (int): index

        Returns:
            (np.array): source image with size (2 * 96, 2 * 96, 3) with values between 0 and 1
            (np.array): destination image with size (2 * 96, 2 * 96, 3) with values between 0 and 1
        """
        image_file_src = choice(self.image_files_src)
        image_file_dst = choice(self.image_files_dst)
        image_src = np.asarray(Image.open(image_file_src).resize((2 * 96, 2 * 96))) / 255.
        image_dst = np.asarray(Image.open(image_file_dst).resize((2 * 96, 2 * 96))) / 255.

        return image_src, image_dst

    def collate_fn(self, batch):
        """
        Collate function to arrange the data returns from a batch. The batch returns a list
        of tuples contains pairs of source and destination images, which is the input of this
        function, and the function returns a tuple with 4 4D tensors of the warp and target
        images for the source and destination
        Args:
            batch (list): a list of tuples contains pairs of source and destination images
                as numpy array

        Returns:
            (torch.Tensor): a 4D tensor of the wrap version of the source images
            (torch.Tensor): a 4D tensor of the target source images
            (torch.Tensor): a 4D tensor of the wrap version of the destination images
            (torch.Tensor): a 4D tensor of the target destination images
        """
        images_src, images_dst = list(zip(*batch))  # convert list of tuples with pairs of images into tuples of source and destination images
        warp_image_src, target_image_src = get_training_data(images_src, len(images_src))
        warp_image_src = torch.tensor(warp_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_src = torch.tensor(target_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        warp_image_dst, target_image_dst = get_training_data(images_dst, len(images_dst))
        warp_image_dst = torch.tensor(warp_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_dst = torch.tensor(target_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        return warp_image_src, target_image_src, warp_image_dst, target_image_dst


def pixel_norm(x, dim=-1):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + 1e-06)


def depth_to_space(x, size=2):
    """
    Upscaling method that use the depth dimension to
    upscale the spatial dimensions
    Args:
        x (torch.Tensor): a tensor to upscale
        size (float): the scaling factor

    Returns:
        (torch.Tensor): new spatial upscale tensor
    """
    b, c, h, w = x.shape
    out_h = size * h
    out_w = size * w
    out_c = c // (size * size)
    x = x.reshape((-1, size, size, out_c, h, w))
    x = x.permute((0, 3, 4, 1, 5, 2))
    x = x.reshape((-1, out_c, out_h, out_w))
    return x


class DepthToSpace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, size=2):
        return depth_to_space(x, size)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.Flatten(),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = pixel_norm(x, dim=-1)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.LeakyReLU(0.1, True),
            DepthToSpace()
        )

    def forward(self, x):
        x = self.upsample(x)
        # x = depth_to_space(x, 2)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

    def forward(self, input):
        x = self.conv1(input)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = x + input
        x = nn.functional.leaky_relu(x, 0.2)
        return x

class Inter(nn.Module):
    def __init__(self):
        super().__init__()
        self.inter = nn.Sequential(
            nn.Linear(18432, 128),
            nn.Linear(128, 1152),
            nn.Unflatten(1, (128, 3, 3)),
            Upsample(128, 512)
        )

    def forward(self, x):
        x = self.inter(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            Upsample(128, 2048),
            ResBlock(512),
            Upsample(512, 1024),
            ResBlock(256),
            Upsample(256, 512),
            ResBlock(128)
        )
        self.conv_out = nn.Conv2d(128, 3, 1, padding='same')
        self.conv_out1 = nn.Conv2d(128, 3, 3, padding='same')
        self.conv_out2 = nn.Conv2d(128, 3, 3, padding='same')
        self.conv_out3 = nn.Conv2d(128, 3, 3, padding='same')
        self.depth_to_space = DepthToSpace()

    def forward(self, x):
        x = self.decoder(x)
        out = self.conv_out(x)
        out1 = self.conv_out1(x)
        out2 = self.conv_out2(x)
        out3 = self.conv_out3(x)
        x = torch.concat((out, out1, out2, out3), 1)
        x = self.depth_to_space(x, 2)
        x = nn.functional.sigmoid(x)
        return x


def create_window(size=11, sigma=1.5, channels=1):
    """
    Create a 2D Gaussian window
    Args:
        size (int): size of the window
        sigma (float): Gaussian standard deviation
        channels (int): number of channels

    Returns:
        (torch.Tensor): Gaussian window of size (channels, 1, size, size)
    """
    gaussian_kernel_1d = torch.tensor(cv2.getGaussianKernel(size, sigma), dtype=torch.float32)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.t()
    gaussian_kernel_2d = gaussian_kernel_2d.expand((channels, 1, size, size)).contiguous().clone()
    return gaussian_kernel_2d

def dssim(image1, image2, window_size=11):
    """
    Calculate the structural dis-similarity index measure. Can read more
    here: https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
    Args:
        image1 (torch.Tensor): image tensor
        image2 (torch.Tensor): image tensor
        window_size (int): the size of the windows to use

    Returns:
        (float) the dssim score
    """
    pad = window_size // 2
    window = create_window(window_size, channels=3).to(device)

    # calculate the local means of the images (luminosity params)
    mu1 = nn.functional.conv2d(image1, window, padding=pad, groups=3)
    mu2 = nn.functional.conv2d(image2, window, padding=pad, groups=3)

    mu1_squared = mu1 ** 2
    mu2_squared = mu2 ** 2
    mu12 = mu1 * mu2

    # calculate the local variance of the images (contrast params)
    sigma1_squared = nn.functional.conv2d(image1 * image1, window, padding=pad, groups=3) - mu1_squared + 1e-6
    sigma2_squared = nn.functional.conv2d(image2 * image2, window, padding=pad, groups=3) - mu2_squared + 1e-6
    sigma12 =  nn.functional.conv2d(image1 * image2, window, padding=pad, groups=3) - mu12 + 1e-6

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminosity_metric = (2 * mu12 + C1) / (mu1_squared + mu2_squared + C1)
    contrast_metric = (2 * torch.sqrt(sigma1_squared * sigma2_squared) + C2) / (sigma1_squared + sigma2_squared + C2)
    structer_metric = (sigma12 + C3) / (torch.sqrt(sigma1_squared * sigma2_squared) + C3)

    ssim = luminosity_metric * contrast_metric * structer_metric
    dssim = (1 - ssim.mean()) / 2

    return dssim

def draw_results(reconstruct_src, target_src, reconstruct_dst, target_dst, fake, loss_src, loss_dst):
    fig, axes = plt.subplots(figsize=(660 * px, 370 * px))
    axes.plot(loss_src, label='loss src')
    axes.plot(loss_dst, label='loss dst')
    plt.legend()
    plt.title(f'press q to quit and save, or r to refresh\nepoch = {len(loss_src)}')
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape((height, width, 3)) / 255.

    images_for_grid = []
    for ii in range(3):
        images_for_grid.extend([reconstruct_src[ii], target_src[ii], reconstruct_dst[ii], target_dst[ii], fake[ii]])

    im_grid = torchvision.utils.make_grid(images_for_grid, 5, padding = 30).permute(1,2,0).cpu().numpy()
    final_image = np.vstack([image_array, im_grid])
    final_image = final_image[..., ::-1]  # convert to BGR
    return final_image

def train(data_path: str, model_name='Quick96', new_model=False, saved_models_dir='saved_model'):
    saved_models_dir = Path(saved_models_dir)
    lr = 1e-4
    dataset = FaceData(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    optim_encoder = torch.optim.Adam([{"params": encoder.parameters()}, {"params": inter.parameters()}], lr=lr)
    optim_decoder_src = torch.optim.Adam(decoder_src.parameters(), lr=lr)
    optim_decoder_dst = torch.optim.Adam(decoder_dst.parameters(), lr=lr)
    criterion_L2 = nn.MSELoss()

    if not new_model and (saved_models_dir / f'{model_name}.pth').exists():
        print('Loading previous model...')
        saved_model = torch.load(str(saved_models_dir / f'{model_name}.pth'))
        epoch = saved_model['epoch']
    else:
        saved_model = {}
        epoch = 0
        mean_loss_src = []
        mean_loss_dst = []

    if saved_model:
        print('loading model states')
        encoder.load_state_dict(saved_model['encoder'])
        inter.load_state_dict(saved_model['inter'])
        decoder_src.load_state_dict(saved_model['decoder_src'])
        decoder_dst.load_state_dict(saved_model['decoder_dst'])
        optim_encoder.load_state_dict(saved_model['optimizer_encoder'])
        optim_decoder_src.load_state_dict(saved_model['optimizer_decoder_src'])
        optim_decoder_dst.load_state_dict(saved_model['optimizer_decoder_dst'])
        mean_loss_src = saved_model['mean_loss_src']
        mean_loss_dst = saved_model['mean_loss_dst']

    mean_epoch_loss_src = np.zeros(len(dataloader))
    mean_epoch_loss_dst = np.zeros(len(dataloader))

    encoder.train()
    inter.train()
    decoder_src.train()
    decoder_dst.train()

    first_run = True
    run = True

    print(len(dataloader.dataset))
    print(len(dataloader))
    while run:
        epoch += 1
        for ii, (warp_im_src, target_im_src, warp_im_dst, target_im_dst) in enumerate(dataloader):

            # source image
            latent_src = inter(encoder(warp_im_src))
            reconstruct_im_src = decoder_src(latent_src)
            loss_dssim = dssim(reconstruct_im_src, target_im_src)
            loss_l2 = criterion_L2(reconstruct_im_src, target_im_src)
            loss = loss_dssim + loss_l2
            optim_encoder.zero_grad()
            optim_decoder_src.zero_grad()

            loss.backward()
            optim_encoder.step()
            optim_decoder_src.step()
            loss_src = loss.item()

            # destination image
            latent_dst = inter(encoder(warp_im_dst))
            reconstruct_im_dst = decoder_dst(latent_dst)
            loss_dssim = dssim(reconstruct_im_dst, target_im_dst)
            loss_l2 = criterion_L2(reconstruct_im_dst, target_im_dst)
            loss = loss_dssim + loss_l2
            optim_encoder.zero_grad()
            optim_decoder_dst.zero_grad()
            loss.backward()
            optim_encoder.step()
            optim_decoder_dst.step()
            loss_dst = loss.item()

            # statistics
            mean_epoch_loss_src[ii] = loss_src
            mean_epoch_loss_dst[ii] = loss_dst

            if first_run:
                first_run = False
                plt.ioff()
                fake = decoder_src(inter(encoder(target_im_dst)))
                result_image = draw_results(reconstruct_im_src, target_im_src, reconstruct_im_dst, target_im_dst, fake, mean_loss_src, mean_loss_dst)
                cv2.imshow(f'results', result_image)
                cv2.waitKey(1)

            k = cv2.waitKey(1)
            if k == ord('q'):
                saved_model['epoch'] = epoch
                saved_model['encoder'] = encoder.state_dict()
                saved_model['inter'] = inter.state_dict()
                saved_model['decoder_src'] = decoder_src.state_dict()
                saved_model['decoder_dst'] = decoder_dst.state_dict()
                saved_model['optimizer_encoder'] = optim_encoder.state_dict()
                saved_model['optimizer_decoder_src'] = optim_decoder_src.state_dict()
                saved_model['optimizer_decoder_dst'] = optim_decoder_dst.state_dict()
                saved_model['mean_loss_src'] = mean_loss_src
                saved_model['mean_loss_dst'] = mean_loss_dst
                saved_models_dir.mkdir(exist_ok=True, parents=True)
                torch.save(saved_model, str(saved_models_dir / f'{model_name}.pth'))
                run = False
                break
            elif k == ord('r'):
                fake = decoder_src(inter(encoder(target_im_dst)))
                result_image = draw_results(reconstruct_im_src, target_im_src, reconstruct_im_dst, target_im_dst, fake, mean_loss_src, mean_loss_dst)
                cv2.imshow('results', result_image)
                cv2.waitKey(1)


        mean_loss_src.append(mean_epoch_loss_src.mean())
        mean_loss_dst.append(mean_epoch_loss_dst.mean())
    cv2.destroyAllWindows()


