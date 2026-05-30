"""Build a cleaned, debugged version of the PyTorch Zero-to-Hero notebook.

Strategy: load original notebook, replace the source of every buggy code cell
with a fixed version (keyed by cell index in the original), then write a new
.ipynb. We also strip all existing outputs so the notebook is reproducible.
"""
import json
from pathlib import Path

SRC = Path("Copy of Zero to Hero in Pytorch(ML+DL+RL)...!!!\U0001f44c\U0001f44c\U0001f44c")
DST = Path("Zero_to_Hero_in_PyTorch_CLEAN.ipynb")

with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ---------- helper ----------
def src_lines(text: str):
    """nbformat stores source as list of lines with newline terminators."""
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]

# ---------- replacements keyed by original cell index ----------
REPLACEMENTS: dict[int, str] = {}

# Cell 1 - intro imports. Strip the Kaggle docker boilerplate.
REPLACEMENTS[1] = """# Core scientific stack
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline

import os
import warnings
warnings.filterwarnings("ignore")
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
"""

# Cell 55 - F.sigmoid / F.tanh are deprecated.
REPLACEMENTS[55] = """y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

# y_softmax = F.softmax(x, dim=0)
# softmax is a special kind of activation function, it is about probability
# and will make the sum as 1.
"""

# Cell 66 - Linear-regression loop. Wrap parameter update in torch.no_grad()
# so we don't mutate leaves under autograd.
REPLACEMENTS[66] = """Xt = torch.from_numpy(X).float()
yt = torch.from_numpy(y).float()

for epoch in range(2500):
    # Compute predictions
    y_pred = linear(Xt)

    # Mean-squared error
    loss = torch.mean((y_pred - yt) ** 2)

    # Back-propagation
    loss.backward()

    # Manual SGD update (under no_grad to avoid tracking)
    with torch.no_grad():
        W -= 0.005 * W.grad
        b -= 0.005 * b.grad
        W.grad.zero_()
        b.grad.zero_()

print("Trained W:", W.item(), "| b:", b.item(), "| final loss:", loss.item())
"""

# Cell 78 - regression model fitting loop. Original used .data.numpy() on a
# scalar tensor for plt.text, which is fragile on modern numpy/matplotlib.
REPLACEMENTS[78] = """for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 10 == 0:
        plt.figure(figsize=(20, 6))
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy(), color="orange")
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)
        plt.text(0.3, 0, 'Loss=%.4f' % loss.item(),
                 fontdict={'size': 25, 'color': 'red'})
        plt.show()
"""

# Cell 90 - classification training. F.softmax needs explicit dim.
REPLACEMENTS[90] = """for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0 or t in [3, 6]:
        plt.figure(figsize=(20, 6))
        plt.cla()
        _, prediction = torch.max(F.softmax(out, dim=1), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).sum()) / 200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy,
                 fontdict={'size': 20, 'color': 'red'})
        plt.show()
"""

# Cell 104 - torch.load with weights_only kwarg is only valid in PyTorch >= 2.4.
# Wrap in try/except so older runtimes work too.
REPLACEMENTS[104] = """def restore_net():
    # restore entire net1 to net2
    try:
        net2 = torch.load('net.pkl', weights_only=False)
    except TypeError:
        # older PyTorch where the weights_only kwarg doesn't exist
        net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.figure(1, figsize=(20, 5))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy(), color="orange")
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
"""

# Cell 115 - num_workers=2 crashes on Windows when not under __main__.
# Use 0 for portability across Colab + local.
REPLACEMENTS[115] = """torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # portable across Colab/Linux/Windows
)
"""

# Cell 128 - same num_workers fix.
REPLACEMENTS[128] = """torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
"""

# Cell 137 - optimizer comparison loop. loss.data[0] -> .item().
REPLACEMENTS[137] = """for epoch in range(EPOCH):
    print('Epoch:', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x, b_y = batch_x, batch_y

        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.item())

plt.figure(figsize=(20, 10))
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
"""

# Cell 144 - Hyperparams for CNN/MNIST. Need DOWNLOAD_MNIST = True for Colab.
REPLACEMENTS[144] = """# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = True   # download MNIST if not present
"""

# Cell 146 - MNIST download path. ../input/mnist is Kaggle; use ./mnist for Colab.
REPLACEMENTS[146] = """# MNIST digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
"""

# Cell 148 - train_data.train_data / .train_labels deprecated.
REPLACEMENTS[148] = """# plot one example
print(train_data.data.size())     # torch.Size([60000, 28, 28])
print(train_data.targets.size())  # torch.Size([60000])
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0].item())
plt.show()
"""

# Cell 151 - same .test_data -> .data renames; matching download root.
REPLACEMENTS[151] = """# Test data (no transform needed; we manually scale to [0,1])
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, download=DOWNLOAD_MNIST)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]
"""

# Cell 157 - CNN training. loss.data[0] -> .item(); TSNE n_iter -> max_iter
# (fallback for older sklearn); F.softmax dim is already correct (not used here).
REPLACEMENTS[157] = """# Visualization helper (skipped if scikit-learn isn't installed)
from matplotlib import cm
try:
    from sklearn.manifold import TSNE
    HAS_SK = True
except ImportError:
    HAS_SK = False
    print('Please install scikit-learn for layer visualization.')

def plot_with_labels(lowDWeights, labels):
    plt.figure(figsize=(20, 6))
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x_, y_, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * int(s) / 9))
        plt.text(x_, y_, str(int(s)), backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()

# Training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            print('Epoch:', epoch,
                  '| train loss: %.4f' % loss.item(),
                  '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # sklearn renamed n_iter -> max_iter in 1.5
                try:
                    tsne = TSNE(perplexity=30, n_components=2,
                                init='pca', max_iter=5000)
                except TypeError:
                    tsne = TSNE(perplexity=30, n_components=2,
                                init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
"""

# Cell 163 - RNN-cls hyperparams (already DOWNLOAD_MNIST=True, fine).
# but let's normalize to ./mnist/ path in cell 164:
REPLACEMENTS[164] = """# MNIST digits dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
"""

# Cell 165 - .train_data / .train_labels renames.
REPLACEMENTS[165] = """print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0].item())
plt.show()
"""

# Cell 169 - volatile=True is removed. Use torch.no_grad() context.
# Also fix test_data download flag and use .data / .targets.
REPLACEMENTS[169] = """test_data = dsets.MNIST(root='./mnist/', train=False,
                       transform=transforms.ToTensor(),
                       download=DOWNLOAD_MNIST)
with torch.no_grad():
    test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.   # (2000, 28, 28)
test_y = test_data.targets.numpy().squeeze()[:2000]
"""

# Cell 175 - RNN-cls training. loss.data[0] -> .item(); test_y.size is fine for
# numpy 1-d, but use len() to be explicit.
REPLACEMENTS[175] = """for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        b_y = y

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            with torch.no_grad():
                test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = float((pred_y == test_y).sum()) / len(test_y)
            print('Epoch:', epoch,
                  '| train loss: %.4f' % loss.item(),
                  '| test accuracy: %.2f' % accuracy)
"""

# Cell 177 - same .data.numpy() pattern, no-grad wrap.
REPLACEMENTS[177] = """with torch.no_grad():
    test_output = rnn(test_x[:20].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:20], 'real number')
"""

# Cell 192 - sin-cos RNN training. Already mostly OK; clean up h_state repack
# and make plotting in-place rather than flicker-prone plt.figure each step.
REPLACEMENTS[192] = """for step in range(60):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.detach()  # break the connection from last iteration

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        plt.figure(figsize=(20, 5))
        plt.plot(steps, y_np.flatten(), 'r-', label="Actual")
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-', label="Predicted")
        plt.xlabel("Steps")
        plt.ylabel("Actual/Predicted")
        plt.title("Step %d, loss=%.4f" % (step, loss.item()))
        plt.legend(loc='best')
        plt.show()
"""

# Cell 198 - AutoEncoder MNIST loader. Need download=True + ./mnist/ path.
REPLACEMENTS[198] = """train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
"""

# Cell 199 - .train_data / .train_labels renames.
REPLACEMENTS[199] = """print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[2].item())
plt.show()
"""

# Cell 205 - AutoEncoder training. .train_data rename + loss.data[0] -> .item().
# Also reduce EPOCH-based gating to avoid the bug that 'epoch in [0,5,EPOCH-1]'
# may not include intermediate ones; keep behaviour but use .item().
REPLACEMENTS[205] = """autoencoder = AutoEncoder()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# Original data (first row) for viewing
view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 and epoch in [0, 5, EPOCH - 1]:
            print('Epoch:', epoch, '| train loss: %.4f' % loss.item())

            with torch.no_grad():
                _, decoded_data = autoencoder(view_data)

            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[0][i].set_xticks(()); a[0][i].set_yticks(())
            for i in range(N_TEST_IMG):
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.show()
"""

# Cell 207 - 3D plot. Axes3D(fig) is deprecated; use fig.add_subplot.
REPLACEMENTS[207] = """# Visualize in 3D plot
view_data = train_data.data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
with torch.no_grad():
    encoded_data, _ = autoencoder(view_data)

fig = plt.figure(2, figsize=(15, 6))
ax = fig.add_subplot(111, projection='3d')
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()
values = train_data.targets[:200].numpy()
for x_, y_, z_, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255 * int(s) / 9))
    ax.text(x_, y_, z_, str(int(s)), backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
"""

# Cell 209 - DQN imports. Add gym wrappers.
REPLACEMENTS[209] = """import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
"""

# Cell 211 - CartPole-v0 was removed. Use v1 and read N_STATES robustly.
REPLACEMENTS[211] = """# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

env = gym.make('CartPole-v1')
# Older gym used env = env.unwrapped to bypass episode-step limits; modern gym keeps
# the same API but we still call unwrapped to access env.x_threshold etc.
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
print('N_ACTIONS:', N_ACTIONS, '| N_STATES:', N_STATES)
"""

# Cell 215 - DQN class. Multiple bugs to fix:
#   * choose_action: torch.max returns shape (1,), .numpy()[0,0] is wrong.
#   * learn: q_target shape mismatch with q_eval.
REPLACEMENTS[215] = """class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:           # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:                                       # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s  = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a  = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r  = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)                  # (batch, 1)
        q_next = self.target_net(b_s_).detach()                     # don't backprop
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
"""

# Cell 217 - Originally a string literal commented-out training loop with the
# OLD gym API (env.reset() -> obs, env.step() -> 4-tuple, env.render()).
# Activate it and adapt to the gym >= 0.26 5-tuple API.
REPLACEMENTS[217] = """print('Collecting experience...')

def reset_env(env):
    out = env.reset()
    # gym >= 0.26 returns (obs, info); older returns obs only
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        s_, r, terminated, truncated, info = out
        done = terminated or truncated
    else:
        s_, r, done, info = out
    return s_, r, done, info

# Train for a small number of episodes so the cell finishes in Colab without
# needing GPU or a display. Bump N_EPISODES for serious training.
N_EPISODES = 50
for i_episode in range(N_EPISODES):
    s = reset_env(env)
    ep_r = 0
    while True:
        a = dqn.choose_action(s)
        s_, r, done, info = step_env(env, a)

        # shape the reward as the original tutorial did
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)
        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep:', i_episode, '| Ep_r:', round(ep_r, 2))

        if done:
            break
        s = s_

env.close()
"""

# Cell 226 - basic GAN training loop.
# Bug in original: opt_D.step() mutates D's params in-place, so the *next*
# G_loss.backward() (which still references the pre-step params via the shared
# graph) fails in PyTorch >= 1.5 with "modified by an inplace operation".
# Fix: detach G_paintings when training D, and re-run G_paintings through D
# for the G update so it gets a fresh, intact graph.
REPLACEMENTS[226] = """for step in range(800):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = G(G_ideas)

    # ---- Train D ----
    prob_artist0 = D(artist_paintings)              # D wants this ~1
    prob_artist1 = D(G_paintings.detach())          # detach -> no grad into G
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward()
    opt_D.step()

    # ---- Train G (fresh forward through the now-updated D) ----
    prob_artist1_for_G = D(G_paintings)             # graph intact for G
    G_loss = torch.mean(torch.log(1. - prob_artist1_for_G))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 100 == 0:
        plt.figure(figsize=(20, 5))
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0],
                 c='#4AD631', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1,
                 c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0,
                 c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3,
                 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2.0,
                 'D score=%.2f (-1.38 for G to converge)' % -D_loss.item(),
                 fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.show()
"""

# Cell 236 - CGAN transform pipeline. Original used 3-channel mean/std but
# MNIST is single-channel; Normalize then errors with broadcast-shape mismatch.
REPLACEMENTS[236] = """transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # MNIST is 1-channel
])
"""

# Cell 228 - CGAN imports. scipy.misc was removed in scipy >= 1.3. Use PIL.
REPLACEMENTS[228] = """import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
from PIL import Image  # used to save sample images (scipy.misc.imsave is gone)
"""

# Cell 231 - to_onehot. The original isinstance check on torch.LongTensor is
# fragile; use dtype-based check. Also keep CPU/CUDA agnostic.
REPLACEMENTS[231] = """def to_onehot(x, num_classes=10):
    if isinstance(x, int):
        c = torch.zeros(1, num_classes).long()
        c[0][x] = 1
        return c
    if torch.is_tensor(x):
        x_cpu = x.detach().cpu().long()
        if x_cpu.dim() == 1:
            x_cpu = x_cpu.view(-1, 1)
        c = torch.zeros(x_cpu.size(0), num_classes).long()
        c.scatter_(1, x_cpu, 1)
        return c
    raise TypeError("to_onehot expects int or torch tensor, got %s" % type(x))
"""

# Cell 232 - get_sample_image. Detach + cpu, and don't reference line_img
# before assignment.
REPLACEMENTS[232] = """def get_sample_image(G, n_noise=100):
    \"\"\"Build a 10x10 grid of generated samples, one row per digit class.\"\"\"
    all_img = None
    for num in range(10):
        c = to_cuda(to_onehot(num))
        line_img = None
        for i in range(10):
            z = to_cuda(torch.randn(1, n_noise))
            y_hat = G(z, c).view(28, 28)
            line_img = y_hat if line_img is None else torch.cat((line_img, y_hat), dim=1)
        all_img = line_img if all_img is None else torch.cat((all_img, line_img), dim=0)
    img = all_img.detach().cpu().numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)  # Tanh output is [-1, 1]
    return img
"""

# Cell 241 - lower max_epoch so the cell finishes in a reasonable time when
# users hit Run-All. Bump it for real training.
REPLACEMENTS[241] = """max_epoch = 3  # bump (e.g. 100-200) for serious training
step = 0
n_critic = 5
n_noise = 100
"""

# Cell 242 - shape mismatch with D output (D returns (batch, 1) but labels
# were (batch,) before — gives a BCELoss warning). Match shapes.
REPLACEMENTS[242] = """D_labels = to_cuda(torch.ones(batch_size, 1))   # real
D_fakes  = to_cuda(torch.zeros(batch_size, 1))  # fake
"""

# Cell 243 - CGAN training loop. Fix: loss.data[0] -> .item(); save image using
# PIL instead of scipy.misc.imsave; sample-saving block was indented inside
# the inner for-idx loop, so it ran every batch -> move out + gate properly.
REPLACEMENTS[243] = """for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        step += 1
        # ---- Train Discriminator ----
        x = to_cuda(images)
        y = labels.view(batch_size, 1)
        y = to_cuda(to_onehot(y)).float()
        x_outputs = D(x, y)
        D_x_loss = criterion(x_outputs, D_labels)

        z = to_cuda(torch.randn(batch_size, n_noise))
        z_outputs = D(G(z, y), y)
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        # ---- Train Generator every n_critic steps ----
        if step % n_critic == 0:
            z = to_cuda(torch.randn(batch_size, n_noise))
            z_outputs = D(G(z, y), y)
            G_loss = criterion(z_outputs, D_labels)

            G.zero_grad()
            G_loss.backward()
            G_opt.step()

        if step % 200 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, G Loss: {:.4f}'.format(
                epoch, max_epoch, step, D_loss.item(), G_loss.item()))

    # Save a sample once per epoch (epoch loop, not the inner one).
    G.eval()
    with torch.no_grad():
        img = get_sample_image(G)
    Image.fromarray(img).save('{}_epoch_{}.jpg'.format(MODEL_NAME, epoch))
    G.train()
"""


# ---------- apply replacements ----------
applied = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    cell["outputs"] = []
    cell["execution_count"] = None
    if i in REPLACEMENTS:
        cell["source"] = src_lines(REPLACEMENTS[i].rstrip("\n"))
        applied += 1

# Update notebook-level metadata so nbformat >= 4.4 doesn't squawk.
nb["nbformat"] = 4
nb["nbformat_minor"] = 5
nb.setdefault("metadata", {}).setdefault("kernelspec", {
    "display_name": "Python 3", "language": "python", "name": "python3",
})
nb["metadata"]["language_info"] = {
    "name": "python", "version": "3.10", "mimetype": "text/x-python",
    "file_extension": ".py", "pygments_lexer": "ipython3",
}

# Prepend a Colab/local setup cell so users get the right deps on the first run.
setup_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": src_lines(
        "# Zero to Hero in PyTorch — Cleaned & Debugged Edition\n"
        "\n"
        "This is a debugged version of the original notebook. The cell directly\n"
        "below installs the dependencies needed by every section (gym is needed\n"
        "for the DQN section; pip-installs are no-ops if already present).\n"
    ),
}
setup_code = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": src_lines(
        "# --- environment setup (safe to re-run) ---\n"
        "import sys, subprocess\n"
        "def _pip(args):\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])\n"
        "try:\n"
        "    import gym  # noqa: F401\n"
        "except ImportError:\n"
        "    _pip(['gym==0.26.2'])\n"
        "try:\n"
        "    import seaborn  # noqa: F401\n"
        "except ImportError:\n"
        "    _pip(['seaborn'])\n"
        "import torch, torchvision\n"
        "print('torch:', torch.__version__, '| torchvision:', torchvision.__version__,\n"
        "      '| cuda:', torch.cuda.is_available())\n"
    ),
}
nb["cells"] = [setup_md, setup_code] + nb["cells"]

with DST.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Applied {applied} replacements. Wrote {DST}.")
print(f"Total cells in cleaned notebook: {len(nb['cells'])}")
