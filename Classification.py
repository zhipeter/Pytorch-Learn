import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)
plt.ion()
plt.figure(figsize=(16, 8), dpi=80)
plt.subplot(121)
plt.scatter(
    x.data.numpy()[:, 0],
    x.data.numpy()[:, 1],
    c=y.data.numpy(),
    s=100,
    lw=0,
    cmap='RdYlGn')


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)  # define the network
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.CrossEntropyLoss()

plt.subplot(122)
for t in range(20):
    out = net.forward(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(
            x.data.numpy()[:, 0],
            x.data.numpy()[:, 1],
            c=pred_y,
            s=100,
            lw=0,
            cmap='RdYlGn')
        accuracy = float(
            (pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(
            1.5,
            -4,
            'Accuracy=%.2f' % accuracy,
            fontdict={
                'size': 20,
                'color': 'red'
            })
        plt.text(1.5, -3, t, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
