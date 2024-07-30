import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchinfo import summary
import utils


BATCH_SIZE = 128
INPUT_SIZE = (BATCH_SIZE, 3, 32, 32)
LEARNING_RATE = 0.01
EPOCHS = 50
DATA_AUGMENTATION = [
    transforms.RandomCrop(INPUT_SIZE[2:], padding=4),
    transforms.RandomErasing(p=0.5, scale=(
        0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.RandomHorizontalFlip(p=0.5)
]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用するモジュールの定義
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(256*2*2, 10)

    def forward(self, x):
        # 順伝播
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.pool(x)

        x = self.activation(self.bn7(self.conv7(x)))
        x = self.activation(self.bn8(self.conv8(x)))
        x = self.pool(x)

        x = x.view(-1, 256*2*2)
        x = self.fc1(x)
        return x


def train(model, device, train_loader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    correct = 0
    for data, target in tqdm(train_loader, desc='Train\t', leave=False, ncols=100):
        data, target = data.to(device), target.to(device)
        # 勾配の初期化
        optimizer.zero_grad()
        # 順伝播
        output = model(data)
        # 損失の計算
        loss = loss_fn(output, target)
        train_loss += loss.item()
        # 逆伝播
        loss.backward()
        # 重みの更新
        optimizer.step()
        # 正解数の計算
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, train_accuracy


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader, desc='Test\t', leave=False, ncols=100):
        data, target = data.to(device), target.to(device)
        # 順伝播
        output = model(data)
        # 損失の計算
        test_loss += loss_fn(output, target).item()
        # 正解数の計算
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    # データセット
    train_loader, test_loader = utils.dataset(INPUT_SIZE, data_augmentation=DATA_AUGMENTATION)
    # モデル
    model = MyModel().to(device)
    # 最適化手法
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,momentum=0.9, weight_decay=5e-4)
    # 損失関数
    loss_fn = nn.CrossEntropyLoss()
    # 学習率のスケジューラ
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)

    summary(model, INPUT_SIZE, device=device)
    # 学習ループ
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epoch\t', ncols=100):
        # 学習
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, loss_fn)
        # テスト
        test_loss, test_accuracy = test(model, device, test_loader, loss_fn)
        # スケジューラの更新
        scheduler.step()
        # ログの記録
        logger.record(epoch, train_loss, test_loss,train_accuracy, test_accuracy)
    # 結果の保存
    folder_name = 'cifar10'
    utils.save(folder_name, logger, model, device, INPUT_SIZE, hyperparams={
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'data_augmentation': DATA_AUGMENTATION,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'scheduler': scheduler,
    })


if __name__ == '__main__':
    # 環境設定・初期化等
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = utils.Logger()
    main()
