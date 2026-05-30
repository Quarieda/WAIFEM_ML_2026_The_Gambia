"""Run just the GAN and CGAN code paths to verify the two fixes work.

These are the two cells that failed in the first end-to-end run. We execute
their code (with reduced iteration counts) and report success/failure.
"""
import sys
import traceback


def run_gan():
    import torch
    import torch.nn as nn
    import numpy as np

    torch.manual_seed(1)
    np.random.seed(1)

    BATCH_SIZE = 64
    LR_G = 0.0001
    LR_D = 0.0001
    N_IDEAS = 5
    ART_COMPONENTS = 15
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

    def artist_works():
        a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
        paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
        return torch.from_numpy(paintings).float()

    G = nn.Sequential(nn.Linear(N_IDEAS, 128), nn.ReLU(), nn.Linear(128, ART_COMPONENTS))
    D = nn.Sequential(nn.Linear(ART_COMPONENTS, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

    for step in range(20):  # small loop, just verifying no crash
        artist_paintings = artist_works()
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
        G_paintings = G(G_ideas)

        # ---- Train D ----
        prob_artist0 = D(artist_paintings)
        prob_artist1 = D(G_paintings.detach())
        D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # ---- Train G ----
        prob_artist1_for_G = D(G_paintings)
        G_loss = torch.mean(torch.log(1. - prob_artist1_for_G))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

    return f"GAN OK after 20 steps. D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}"


def run_cgan():
    import torch
    import torchvision
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    from PIL import Image

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def to_cuda(x):
        return x.to(DEVICE)

    def to_onehot(x, num_classes=10):
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
        raise TypeError(type(x))

    class Discriminator(nn.Module):
        def __init__(self, input_size=784, label_size=10, num_classes=1):
            super().__init__()
            self.l1 = nn.Sequential(nn.Linear(input_size + label_size, 200), nn.ReLU(), nn.Dropout())
            self.l2 = nn.Sequential(nn.Linear(200, 200), nn.ReLU(), nn.Dropout())
            self.l3 = nn.Sequential(nn.Linear(200, num_classes), nn.Sigmoid())

        def forward(self, x, y):
            x, y = x.view(x.size(0), -1), y.view(y.size(0), -1).float()
            v = torch.cat((x, y), 1)
            return self.l3(self.l2(self.l1(v)))

    class Generator(nn.Module):
        def __init__(self, input_size=100, label_size=10, num_classes=784):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(input_size + label_size, 200), nn.LeakyReLU(0.2),
                nn.Linear(200, 200), nn.LeakyReLU(0.2),
                nn.Linear(200, num_classes), nn.Tanh(),
            )

        def forward(self, x, y):
            x, y = x.view(x.size(0), -1), y.view(y.size(0), -1).float()
            v = torch.cat((x, y), 1)
            return self.layer(v).view(x.size(0), 1, 28, 28)

    D = to_cuda(Discriminator())
    G = to_cuda(Generator())

    # The crucial fix: single-channel Normalize for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    mnist = datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)

    batch_size = 64
    data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True)
    criterion = nn.BCELoss()
    D_opt = torch.optim.Adam(D.parameters())
    G_opt = torch.optim.Adam(G.parameters())
    D_labels = to_cuda(torch.ones(batch_size, 1))
    D_fakes = to_cuda(torch.zeros(batch_size, 1))

    n_critic, n_noise, step = 5, 100, 0
    # Only do a handful of batches just to prove the pipeline works.
    for idx, (images, labels) in enumerate(data_loader):
        if idx >= 5:
            break
        step += 1
        x = to_cuda(images)
        y = labels.view(batch_size, 1)
        y = to_cuda(to_onehot(y)).float()
        x_outputs = D(x, y)
        D_x_loss = criterion(x_outputs, D_labels)

        z = to_cuda(torch.randn(batch_size, n_noise))
        z_outputs = D(G(z, y), y)
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss
        D.zero_grad(); D_loss.backward(); D_opt.step()

        if step % n_critic == 0:
            z = to_cuda(torch.randn(batch_size, n_noise))
            z_outputs = D(G(z, y), y)
            G_loss = criterion(z_outputs, D_labels)
            G.zero_grad(); G_loss.backward(); G_opt.step()

    # Verify get_sample_image style code path + PIL save works.
    def get_sample_image(G, n_noise=100):
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
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return img

    G.eval()
    with torch.no_grad():
        img = get_sample_image(G)
    Image.fromarray(img).save('cgan_sample_test.jpg')
    G.train()

    return f"CGAN OK. Image shape={img.shape}, D_loss={D_loss.item():.4f}"


for name, fn in [("GAN", run_gan), ("CGAN", run_cgan)]:
    try:
        msg = fn()
        print(f"[PASS] {name}: {msg}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
