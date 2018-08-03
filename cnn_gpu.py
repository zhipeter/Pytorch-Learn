import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = torch.unsqueeze(
    test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda() / 255
test_y = test_data.test_labels[:2000].cuda()
print(test_y.size(0))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# def plot_with_labels(lowDWeights, lables):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, lables):
#         c = cm.rainbow(int(255 * s / 9))
#         plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max())
#     plt.ylim(Y.min(), Y.max())
#     plt.show()
#     plt.pause(0.01)

# plt.ion()
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x=x.cuda()
        b_y=y.cuda()

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = torch.sum(pred_y == test_y)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data,
                  '| test accuracy: %.2f' % accuracy)

            # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:500, : ])
            # labels = test_y.numpy()[:500]
            # plot_with_labels(low_dim_embs, labels)

# plt.ioff()

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
