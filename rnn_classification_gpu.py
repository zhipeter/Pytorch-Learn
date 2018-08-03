import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = datasets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = datasets.MNIST(
    root='./mnist', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)[:2000].cuda() / 255
test_y = test_data.test_labels[:2000].cuda()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda()
        b_y = y.cuda()
        b_x = b_x.view(-1, 28, 28)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = torch.sum(pred_y == test_y)
            print('Epoch: ', epoch, '| Step: ', step,
                  '| train loss: %.4f' % loss.data.squeeze(),
                  '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
