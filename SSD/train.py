import time
from glob import glob
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from ssd_network import SSD, SSDLoss
from dataset import PascalVOCDataset, label_map, create_data_lists


data_folder = './Data/'
keep_difficult = True
checkpoint = './checkpoint_ssd300.pt'  # path to model checkpoint to load, None if you don't want to load checkpoint

#  Model parameters
num_classes = len(label_map)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#  Learning parameters
batch_size = 4
iterations = 120_000
workers = 4
print_freq = 200
lr = 1e-3
decay_lr_at_iteration = [80_000, 100_000]
decay_lr_factor = 0.1
momentum = 0.9
weight_decay = 5e-4

cudnn.benchmark = True


class AverageMeter:
    """
    Class to track of average, sum and count of metric
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, factor):
    """
    Scale the learning rate by factor
    Args:
        optimizer (torch.optim.Optimizer): optimizer for which the learning rate will be adjusted
        factor (float): the factor the learning rate will be adjusted by
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor
    print(f'Decaying learning rate.\n The new learning rate is {optimizer.param_groups[1]["lr"]}\n')

def save_checkpoint(epoch, model, optimizer):
    """
    A function to save the current state of the training
    Args:
        epoch (int): the current epoch number
        model (torch.nn.Model): the model to save it state
        optimizer (torch.optim.Optimizer): the optimizer to save it state

    Returns:
        None
    """
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    filename = 'checkpoint_ssd300.pt'
    torch.save(state, filename)

def train(train_dataloader, model, criterion, optimizer, epoch):
    """
    The training step in over one epoch
    Args:
        train_dataloader (torch.utils.data.DataLoader): the training dataloader
        model (torch.nn.Module): the network to train
        criterion (torch.nn.modules.loss._Loss): the loss function
        optimizer (torch.optim.Optimizer): the optimizer to use while training the network
        epoch (int): the current epoch number

    Returns:
        None
    """
    model.train()

    losses = AverageMeter()  # track the loss
    batch_time = AverageMeter()  # track the forward + backward propagation time
    data_time = AverageMeter()  # track the data loading time

    start = time.time()
    for ii, (images, boxes, labels, _) in enumerate(train_dataloader):
        data_time.update(time.time() - start)

        # move data to the device
        images = images.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        predicted_locations, predicted_classes_scores = model(images)  # get prediction from the network
        loss = criterion(predicted_locations, predicted_classes_scores, boxes, labels)  # compute loss

        # update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.shape[0])
        batch_time.update(time.time() - start)

        start = time.time()

        if ii % print_freq == 0:
            print(f'Epoch: [{epoch}][{ii}/{len(train_dataloader)}]\t'
                  f'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

        del predicted_locations, predicted_classes_scores, images, boxes, labels  # delete to save memory

def main():
    """
    The main function to run for training the network
    Returns:
        None
    """
    global label_map, decay_lr_at_iteration, checkpoint

    model = SSD(num_classes=num_classes)  # create the network

    start_epoch = 0

    # separate the bias parameter from the non-bias one, so we can set a
    # different learning rate to each category
    biases = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    # initialize the optimizer
    optimizer = optim.SGD(params=[{'params': biases, 'lr':2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    # load checkpoint if exists
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        print(f'\nLoading checkpoint from epoch {start_epoch}\n')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    model = model.to(device)
    criterion = SSDLoss(priors_cx_cy=model.prior_cx_cy).to(device)  # set the loss funtion

    # create tarining set and training dataloader
    train_dataset = PascalVOCDataset(data_folder, data_type='train', keep_difficult=keep_difficult)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)

    # convert iteration to epoch
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at_epoch = [iteration // (len(train_dataset) // batch_size) for iteration in decay_lr_at_iteration]

    for epoch in range(start_epoch, epochs):
        # check if need to decay the learning rate
        if epoch in decay_lr_at_epoch:
            adjust_learning_rate(optimizer, decay_lr_factor)

        # perform training step
        train(train_dataloader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

        save_checkpoint(epoch, model, optimizer)


if __name__ == '__main__':
    voc07_path = './Data/VOC2007'
    voc12_path = './Data/VOC2012'
    output_folder = './Data'
    if not glob(output_folder + '/*.json'):
        create_data_lists(voc07_path, voc12_path, output_folder)
    main()