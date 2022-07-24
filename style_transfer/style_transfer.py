from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from torch.nn import functional as F
from torch import optim
from torchvision.models import vgg19
import matplotlib.pyplot as plt
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocessing = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Lambda(lambda x: x.mul_(255))
])
postprocessing = T.Compose([
    T.Lambda(lambda x: x.mul_(1./255)),
    T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
])


class GramMatrix(nn.Module):
    def forward(self, input):
        batch_size, num_channels, height, width = input.size()
        features_map = input.view(batch_size, num_channels, height * width)
        G_mat = features_map @ features_map.transpose(1, 2)
        G_mat.div_(height * width)
        return G_mat

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = F.mse_loss((GramMatrix()(input)), target)
        return out


class vgg19_modified(nn.Module):
    def __init__(self):
        super().__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
    def forward(self, x, layers=[]):
        order = np.argsort(layers)
        _results, results = [], []
        for ix, model in enumerate(self.features):
            x = model(x)
            if ix in layers:
                _results.append(x)
        for ii in order:
            results.append(_results[ii])
        return results if layers else x


vgg = vgg19_modified().to(device)
img_shape = plt.imread('content.jpg').shape[:2][::-1]
imgs = [Image.open(path).resize((512, 512)).convert('RGB') for path in ['style.jpg', 'content.jpg']]
style_im, content_im = [preprocessing(img).to(device)[None] for img in imgs]

opt_img = content_im.data.clone()
opt_img.requires_grad = True

style_layers = [0, 5, 10, 19, 28]
content_layer = [21]
loss_layers = style_layers + content_layer

loss_func = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layer)
loss_func = [loss.to(device) for loss in loss_func]

style_weights = [1000 / n**2 for n in [64, 128, 256, 512, 512]]
content_weight = [100]
weights = style_weights + content_weight

style_targets = [GramMatrix()(A).detach() for A in vgg(style_im, style_layers)]
content_targets = [A.detach() for A in vgg(content_im, content_layer)]
targets = style_targets + content_targets

max_iters = 500
optimizer = optim.LBFGS([opt_img])

iter_counter = 0
losses = []
test_counter = 0


def closure():
    global iter_counter, losses
    iter_counter = iter_counter + 1
    print(f'{iter_counter + 1}/{max_iters}')
    optimizer.zero_grad()
    out = vgg(opt_img, loss_layers)
    layers_losses = [weights[a] * loss_func[a](A, targets[a]) for a, A in enumerate(out)]
    loss = sum(layers_losses)
    losses.append(loss.item())

    loss.backward()
    return loss


while iter_counter < max_iters:
    optimizer.step(closure)
    iter_counter += 1



with torch.no_grad():
    out_img = postprocessing(opt_img[0]).permute(1, 2, 0).cpu().numpy()
new_image = cv2.resize(out_img, img_shape)
new_image = (new_image - new_image.min(axis=(0,1))) / (new_image.max(axis=(0,1)) - new_image.min(axis=(0,1)))
plt.imshow(new_image)


