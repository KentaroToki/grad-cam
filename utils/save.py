import utils
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn


def save(folder_name, log: utils.Logger, model, device, input_size, hyperparams={}):
    # 保存先フォルダを作成
    root = './result/' + \
        str(datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M_')) + folder_name + '/'
    os.makedirs(root, exist_ok=True)

    # ハイパーパラメータを保存
    with open(root+'hyperparameters.txt', mode='w') as f:
        for key, value in hyperparams.items():
            f.write(f'{key}: {value}\n')

    # ログを保存
    df = pd.DataFrame(log.log)
    df.to_csv(root+'log.csv', index=False)

    # モデル構造を保存
    with open(root+'summary.txt', mode='w') as f:
        f.write(repr(summary(model, (64, 3, 32, 32), device=device, verbose=0)))

    # 損失のグラフを保存
    plt.plot(log.log['epoch'], log.log['train_loss'], label='train_loss')
    plt.plot(log.log['epoch'], log.log['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(root+'loss.png')
    plt.close()

    # 精度のグラフを保存
    plt.plot(log.log['epoch'], log.log['train_accuracy'],
             label='train_accuracy')
    plt.plot(log.log['epoch'], log.log['test_accuracy'], label='test_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(root+'accuracy.png')
    plt.close()

    # 画像と予測ラベルを保存
    N = 4
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    images = []  # (image, label, pred)
    _, test_loader = utils.dataset(input_size=(1,)+input_size[1:], download=False)
    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            label = label.item()
            if label == len(images) % 10:
                output = model(img)
                pred = output.argmax(dim=1, keepdim=True)
                images.append((img.squeeze().permute(
                    1, 2, 0).cpu().numpy(), label, pred.item()))
            if len(images) == 10 * N:
                break
    _, ax = plt.subplots(N, 10, figsize=(24, 3*N))
    ax = ax.ravel()
    for i in range(10*N):
        img, label, pred = images[i]
        ax[i].imshow(img)
        ax[i].axis('off')
        color = 'black' if pred == label else 'red'
        ax[i].set_title(f'{classes[pred]}', color=color)
    plt.tight_layout()
    plt.savefig(root+'images.png')
    plt.close()

    # GradCAMを保存
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    target_layers = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target_layers.append(m)
    target_layers = [target_layers[-1]]
    cam = GradCAM(
        model=model, target_layers=target_layers
    )
    images = []  # (cam_img, label, pred)
    _, test_loader = utils.dataset(input_size=(1,)+input_size[1:], download=False)
    model.eval()
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        label = label.item()
        if label == len(images) % 10:
            grayscale_cam = cam(input_tensor=img, targets=[
                                ClassifierOutputTarget(label)])
            grayscale_cam = grayscale_cam[0, :]
            cam_img = show_cam_on_image(img.squeeze().permute(
                1, 2, 0).cpu().numpy(), grayscale_cam, use_rgb=True, image_weight=0.5)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            images.append((cam_img, label, pred.item()))
        if len(images) == 10 * N:
            break
    _, ax = plt.subplots(N, 10, figsize=(24, 3*N))
    ax = ax.ravel()
    for i in range(10*N):
        img, label, pred = images[i]
        ax[i].imshow(img)
        ax[i].axis('off')
        color = 'black' if pred == label else 'red'
        ax[i].set_title(f'{classes[pred]}', color=color)
    plt.tight_layout()
    plt.savefig(root+'gradcam.png')
    plt.close()
