from UNet import UNet
from dataset import Taco

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm

# initialisation
INPUT_SIZE = (572, 572)
OUTPUT_SIZE = (388, 388)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    return tuple(zip(*batch))


# uses Taco() to create and return dataloaders
def create_data_loaders(ratio, batch_size, data_augmentation=False, super_categories=False):
    taco = Taco(data_augmentation=data_augmentation, super_categories=super_categories)
    train_count = round(len(taco) * ratio)
    test_count = len(taco) - train_count
    train_split, test_split = random_split(taco, [train_count, test_count])
    train_loader = DataLoader(train_split, batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_split, batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    print("Created data loaders")
    return train_loader, test_loader


def save_net(net, path):
    torch.save(net.state_dict(), path)
    print("Saved net")


def load_net(net, path):
    net.load_state_dict(torch.load(path))
    print("Loaded net")


def invert_binary_mask(mask):
    mask[mask == 1] = -1
    mask[mask == 0] = 1
    mask[mask == -1] = 0
    return mask


def train(net, train_loader, test_loader, batch_size, epochs, class_count, lr, vis, plot):

    # initialisation
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    size = OUTPUT_SIZE[0] * OUTPUT_SIZE[1] * batch_size

    for epoch in range(epochs):

        # training
        net.train()
        training_loss = []
        for images, masks in tqdm(train_loader, f'Epoch {epoch + 1}'):
            optimizer.zero_grad()

            # stacks images and masks for computation
            x = torch.stack(images, dim=0).to(torch.float).to(device)
            target = torch.stack(masks, dim=0).to(torch.long).to(device)

            # forward pass
            prediction = net(x)

            # reshapes so CrossEntropyLoss() can compute the loss
            reshaped_prediction = prediction.permute(0, 2, 3, 1).reshape((size, class_count))
            reshaped_target = target.reshape(size)

            # computes loss
            loss = criterion(reshaped_prediction, reshaped_target)
            # saves loss for evaluation
            training_loss.append(loss)

            # backward pass
            loss.backward()
            optimizer.step()

        # evaluation
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.eval()

            validation_loss = []
            binary_ious = []
            multi_ious = []

            for batch, (images, masks) in tqdm(enumerate(test_loader), f'Validation {epoch + 1}'):
                x = torch.stack(images, dim=0).to(torch.float).to(device)
                target = torch.stack(masks, dim=0).to(torch.long).to(device)

                prediction = net(x)

                reshaped_prediction = prediction.permute(0, 2, 3, 1).reshape((size, class_count))
                reshaped_target = target.reshape(size)

                loss = criterion(reshaped_prediction, reshaped_target)
                prediction = torch.softmax(prediction, 1)

                # calculates metrics
                batch_binary_iou, batch_multi_iou = calculate_IoU(prediction, target, (epoch, batch), vis, plot)

                # saves metrics
                validation_loss.append(loss)
                binary_ious.append(batch_binary_iou)
                multi_ious.append(batch_multi_iou)

            average_training_loss = torch.mean(torch.tensor(training_loss))
            average_validation_loss = torch.mean(torch.tensor(validation_loss))
            average_binary_iou = np.mean(binary_ious)
            average_multi_iou = np.mean(multi_ious)

        print(f"Epoch: {epoch}; Trainingloss: {average_training_loss}; Validationloss: {average_validation_loss}; Binary IoU: {average_binary_iou}; Multi IoU: {average_multi_iou}")

        save_net(net, f"./nets/epoch{epoch}.pth")
        torch.cuda.empty_cache()
    print("Training finished")


def calculate_IoU(prediction, target, id, vis=False, plot=False):
    """
    calculated IoU of prediction and target
    :param prediction: tensor of shape [batch_size, n_classes, 388, 388]
    :param target: tensor of shape [batch_size, 388, 388]
    :param id: only used for naming figures 
    :param plot: save plots or not
    :return: batch_results: numpy-array of shape [batch_size, n_classes] containing -1 for class not in true_mask or IoU of class
            average_precision: average over batch_results with -1 not counted
    """
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    if vis:
        visualize_both_masks(prediction, target)

    # applies threshold to one hot encode the prediction
    threshold = 0.5
    prediction[prediction > threshold] = 1
    prediction[prediction <= threshold] = 0
    number_classes = prediction.shape[1]
    batch_size = target.shape[0]
    batch_ious = []

    # iterates over all items in the given batch
    for item in range(batch_size):
        item_ious = []
        to_plot_pred = []
        to_plot_true = []

        # iterates over all trash classes for the given item
        for i in range(number_classes):
            single_prediction = prediction[item][i]
            item_target = deepcopy(target[item])
            # gets the predictions for other classes out of the way
            item_target[item_target != i] = 5000

            # the two masks intersect where the target has its index i and the prediction is 1
            sum_target_prediction = item_target + single_prediction
            # checks for the entries with the correct sum and adds them up
            intersection = np.sum(sum_target_prediction == i + 1)
            # adds up positive areas and subtracts the double counted intersection
            union = np.sum(item_target == i) + np.sum(single_prediction == 1) - intersection

            # marks indices i that had no actual annotation
            if np.sum(item_target == i) == 0:
                item_ious.append(-1)
            else:
                item_ious.append(intersection / union)
                if plot:
                    to_plot_true.append(item_target)
                    to_plot_pred.append(single_prediction)
        if plot:
            plot_for_iou(to_plot_pred, to_plot_true, id, item)

        batch_ious.append(item_ious)

    batch_ious = np.array(batch_ious)
    # computes the mean IoU for the background class/ binary segmentation
    batch_binary_iou = np.mean(batch_ious, axis=0)[0]
    # computes the mean IoU for all trash classes that had a annotation
    batch_multi_iou = (np.sum(batch_ious) + np.sum(batch_ious == -1)) / np.sum(batch_ious != -1)

    return batch_binary_iou, batch_multi_iou

# for methods for visualisation:

# two methods used for IoU:
def plot_for_iou(to_plot_pred, to_plot_true, id, item):
    length = len(to_plot_pred)
    for i in range(length):
        plt.subplot(length, 2, 2 * i + 1)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.imshow(to_plot_pred[i])
        if i == 0:
            plt.title("prediciton")
        plt.subplot(length, 2, 2 * i + 2)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.imshow(to_plot_true[i])
        if i == 0:
            plt.title("true mask")
    plt.savefig(f"./visualisations/iou/epoch{id[0]}count{id[1]}item{item}")


def visualize_both_masks(mask_pred, mask_true):
    for i in range(mask_true.shape[0]):
        plt.subplot(8, 10, 1)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.imshow(mask_true[i]*255)
        plt.title("true mask")
        for j in range(mask_pred.shape[1]):
            plt.subplot(8, 10, 11 + j)
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.imshow(mask_pred[i][j])
        plt.savefig(f"./visualisations/iou/before_threshold/{i}")


# two methods for general use:
def mask_on_picture_binary(index, net_name, net_classes):
    """
    draws the predicted inverted background class on the original image
    :param index: index of picture you want to mask
    :param net_name: path of pre-trained network that should create the mask
    :param net_classes: number of classes the network was trained with
    :return: 
    """
    taco = Taco()
    ex_image, ann, full_ex_image = taco.get_example(index)
    u_net = UNet(net_classes).to(device)
    load_net(u_net, net_name)
    with torch.no_grad():
        u_net.eval()
        x = torch.unsqueeze(ex_image, 0)
        prediction = u_net(x.to(torch.float).to(device))
        prediction = torch.softmax(prediction, 1)
        w = full_ex_image.shape[1]
        h = full_ex_image.shape[2]
        combination = full_ex_image.permute(0, 1, 2).numpy()

        background_class = prediction[0][0]
        mask_to_visualize = background_class.detach().cpu().numpy()

        threshold = 0.5
        mask_to_visualize[mask_to_visualize > threshold] = 1
        mask_to_visualize[mask_to_visualize <= threshold] = 0
        mask_to_visualize = invert_binary_mask(mask_to_visualize)

        array = np.zeros([h, w, 3], dtype=np.uint8)
        array[:, :] = [127, 0, 0]  # colour choice mask

        array = torch.from_numpy(array).permute(2, 1, 0).numpy()
        mask_to_visualize = Image.fromarray(mask_to_visualize).resize((h, w), resample=0)
        mask_to_visualize = np.array(mask_to_visualize)
        mask_to_visualize = mask_to_visualize * array

        combination += 0.01 * mask_to_visualize

        combination = torch.from_numpy(combination).permute(1, 2, 0).numpy()
        plt.imshow(combination)
        plt.savefig(f"./visualisations/mask_index_visualization/mask_vis_binary_index{index}")


def mask_on_picture_n_class(index, net_name, net_classes):
    """
    draws all predicted classes on one image in different colours
    :param index: index of picture you want to mask
    :param net_name: path of pretrained network that should create the mask
    :param net_classes: number of classes the network was trained with
    :return:
    """
    taco = Taco()
    ex_image, ann, full_ex_image = taco.get_example(index)
    u_net = UNet(net_classes).to(device)
    load_net(u_net, net_name)
    with torch.no_grad():
        u_net.eval()
        x = torch.unsqueeze(ex_image, 0)
        prediction = u_net(x.to(torch.float).to(device))
        prediction = torch.softmax(prediction, 1)
        number_classes = prediction.shape[1]
        w = full_ex_image.shape[1]
        h = full_ex_image.shape[2]
        combination = full_ex_image.permute(0, 1, 2).numpy()
        for i in range(1, number_classes):
            bg = prediction[0][i]
            mask_to_visualize = bg.detach().cpu().numpy()

            mean = np.mean(mask_to_visualize)
            max = np.amax(mask_to_visualize)
            threshold = mean + abs(mean - max)*0.5
            mask_to_visualize[mask_to_visualize > threshold] = 1
            mask_to_visualize[mask_to_visualize <= threshold] = 0

            array = np.zeros([h, w, 3], dtype=np.uint8)
            random_number = np.random.randint(0, 255, 2)
            array[:, :] = [random_number[0], 0, random_number[1]]  # colour choice mask

            array = torch.from_numpy(array).permute(2, 1, 0).numpy()
            mask_to_visualize = Image.fromarray(mask_to_visualize).resize((h, w), resample=0)
            mask_to_visualize = np.array(mask_to_visualize)
            mask_to_visualize = mask_to_visualize * array

            combination += 0.01 * mask_to_visualize

        combination = torch.from_numpy(combination).permute(1, 2, 0).numpy()
        plt.imshow(combination)
        plt.savefig(f"./visualisations/mask_index_visualization/mask_vis_{number_classes}_class_index{index}")



if __name__ == '__main__':
    # example for visualisation of a prediction for an image
    # the index of the image is 30 the path of the net is vis_net_path, the class count is 60
    vis_net_path = 0
    if vis_net_path != 0:
        mask_on_picture_binary(30, vis_net_path, 60)
        mask_on_picture_n_class(30, vis_net_path, 60)

    # initialisation dataloader
    batch_size = 3  # batch_size > 3 requires over 6GB of GPU memory which we did not have
    data_augmentation = False
    super_categories = False

    # initialisation net
    if super_categories:
        class_count = 28 + 1
    else:
        class_count = 60 + 1
    color_channels = 3
    halve = False
    batch_norm = False
    load_a_net = False
    net_path = "./nets/epoch1.pth"

    # initialisation train
    epochs = 100
    lr = 1*10**(-5)
    vis = False
    plot = False

    # run
    train_loader, test_loader = create_data_loaders(0.9, batch_size, data_augmentation, super_categories)
    u_net = UNet(class_count, color_channels, halve, batch_norm).to(device)
    if load_a_net:
        load_net(u_net, net_path)
    train(u_net, train_loader, test_loader, batch_size, epochs, class_count, lr, vis, plot)
