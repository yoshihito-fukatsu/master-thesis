import os
import random
import math
import time
import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchinfo
import numpy as np
import matplotlib.pyplot as plt


## set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

### initial setting ###
trainimgs_dir = 'F:\\user-2\\fukatsu3\\DPinput\\LCN3_256_221026\\DP123_all' #ここで選択しているdirの下に仮のdirをもうひとつ作り、そこに画像を格納する必要がある(DataLoaderの性質上必要)
num_epochs = 1000
save_epoch = 10 # save interval of training models
batch_size = 64 # Square numbers are preferred.
image_size = 256
generate_img_num = 50 # number of generate images

nc = 1 # number of channels in the training images
nz = 100 # size of Z latent vector (1 dimensional)
ngf = 64 # size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
dropout = 0.5 # dropout rate in discriminator
momentum = 0.01 # parameter for batchnormalzation "momentum(pytorch)= 1 - momentum(tensorflow)"
eps = 0.001 # parameter for batchnormalzation
lr = 2e-5 # learning rate for Adam optimizers
beta1 = 0.5 # beta1 hyperparameter for Adam optimizers


exist_ok = True
save_dirname = 'BS'+str(batch_size)+'-nz'+str(nz)+'-epoch'+str(num_epochs)+'-lr'+str(lr)+'-beta'+str(beta1)+'-'+str(datetime.date.today())
os.makedirs('./output',exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname, exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/training_images_a_batch', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/training_progression', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/model', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/model/netG', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/model/netD', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/model/optimizerG', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/model/optimizerD', exist_ok=exist_ok)
os.makedirs('./output/'+save_dirname+'/loss', exist_ok=exist_ok)

with open('./output/'+save_dirname+'/training_images_path.txt', 'w') as f:
    f.write(trainimgs_dir)
with open('./output/'+save_dirname+'/randomseed.txt', 'w') as f:
    f.write(str(manualSeed))


class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),  #ToTensor()でテンソル変換と[0,255]→[0,1]への変換を行う
            transforms.Normalize(mean=0.5,std=0.5) #各画像に対して[0,1]→[-1,1]変換をする。generatorの最後がtanhであり、それに合わせるため。参考：https://teratail.com/questions/280822
        ])

    def __call__(self,img):
        return self.data_transform(img)


## Create the dataset and dataloader
train_dataset = datasets.ImageFolder(root=trainimgs_dir, transform=ImageTransform())
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

trainimg_batch, label = next(iter(train_dataloader)) ## operation check of DataLoader
print('# operation check of DataLoader \ ',trainimg_batch.size())

print('Number of training images : {}'.format(len(train_dataset)))
print('batch size : {}'.format(batch_size))
print('Number of iterations per epoch : {}'.format(len(train_dataloader)))


## Decide which device we want to run on
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('device : ',device)


## check training images
num_show = batch_size # Number of training images to show
plt.imsave('./output/'+save_dirname+'/training_images_a_batch/training_images.png',
            np.transpose(vutils.make_grid(trainimg_batch.to(device)[:num_show], padding=5, nrow=int(math.ceil(math.sqrt(batch_size))),normalize=True).cpu().detach().numpy().copy(),(1,2,0)),
            cmap ='gray')


# custom weights initialization called on ''netG'' and ''netD''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ## input is Z, going into a transposed convolution
            nn.Linear(nz,ngf*8*16*16, bias=False),
            nn.BatchNorm1d(ngf*8*16*16, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Unflatten(1, (ngf*8,16,16)),
            ## state size. '(ngf*8) x 16 x 16'
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf*4) x 32 x 32'
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf*2) x 64 x 64'
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ngf) x128 x 128'
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            ## state size. '(nc) x 256 x 256'
        )

    def forward(self, input):
        return self.main(input)


## create the generator
netG = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):  # If multiple GPUs are available, parallelize.
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init) # randomly initialize all weights in netG to 'mean=0', 'stdev=0.02'.
netG_info = torchinfo.summary(model=netG, input_size=[(batch_size,nz)])


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            ## input is '(nc) x 256 x 256'
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(p=dropout),
            ## state size. '(ndf) x 128 x 128'
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ndf*2) x 64 x 64'
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4, momentum=momentum, eps=eps),
            nn.LeakyReLU(0.3, inplace=True),
            ## state size. '(ndf*4) x 32 x 32'
            nn.Flatten(),
            nn.Linear(ndf*4*32*32, 1, bias=False),
            nn.Sigmoid()
            ## state size. 'slacar'
        )

    def forward(self, input):
        return self.main(input)


## create the discriminator
netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):  # If multiple GPUs are available, parallelize.
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init) # randomly initialize all weights in netG to 'mean=0', 'stdev=0.02'.
netD_info = torchinfo.summary(model=netD, input_size=[(batch_size,nc,image_size,image_size)])


## loss functions and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999),
                        # weight_decay=1e-5
                        )
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999),
                        # weight_decay=1e-5
                        )


### Training Loop ###
img_list = []
G_losses_iter = []
D_losses_iter = []
G_losses_epoch = []
D_losses_epoch = []
D_output_real = []
D_output_fake = []
iters = 0
start_time = time.time()

real_label = 1 #0.9 # https://tatsy.github.io/programming-for-beginners/python/stabilize-gan-training/
fake_label = 0 #0.1

fixed_noise = torch.randn(batch_size, nz, device=device) #generate random noise from normal distribution(mean:0, var:1) for visualization of the progression of the generator

print("Starting Training Loop...")
for epoch in range(1, num_epochs+1):
    start_epoch_time = time.time()

    for i, data in enumerate(train_dataloader, 1):  # enumetare:第一引数のリストやタプルにインデックス番号を付帯させながらfor_loopを回せる。第二引数はインデックスの開始番号
        real_img = data[0].to(device)
        b_size = real_img.size(0)
        real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        fake_labels = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        noise = torch.randn(b_size, nz, device=device) # generate batch of latent vectors; generatorの最初がlinearなので2次元のtensor。最初がtransconv2dならば4次元のtensor。
        # noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_img = netG(noise) # Generate fake image batch with G

        ############################
        ### (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        netD.zero_grad()

        output = netD(real_img).view(-1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()

        output = netD(fake_img.detach()).view(-1) # discriminatorがgeneratorが学んできた計算グラフ（計算の経歴）を知っていてはおかしい。そのため、netDにいれるnetGが生成した画像はdetach()により計算グラフを切り離す。
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        ### (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()

        output = netD(fake_img).view(-1) #ここはgeneratorの学習なので、netGが生成した画像をdetach()しなくていい
        errG = criterion(output, real_labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        ## #output training stats
        if (i == (len(train_dataloader) // 2)) or (i == len(train_dataloader)) :
            print('epoch[%d/%d] iter[%d/%d]  Loss_D: %.4f  Loss_G: %.4f  D(x): %.4f  D(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses_iter.append(errG.item())
        D_losses_iter.append(errD.item())
        D_output_real.append(D_x)
        D_output_fake.append(D_G_z1)

        iters += 1

    G_losses_epoch.append(errG.item())
    D_losses_epoch.append(errD.item())

    ## check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake_img = netG(fixed_noise).detach().cpu()
    plt.imsave('./output/'+save_dirname+'/training_progression/generated_images_epoch_{}.png'.format(epoch),
            np.transpose(vutils.make_grid(fake_img[:num_show], padding=5, nrow=int(math.ceil(math.sqrt(batch_size))), normalize=True).cpu().numpy().copy(),(1,2,0)),
            cmap ='gray')



    if epoch % save_epoch == 0:  # save checkpoints
        torch.save(netG.state_dict(), './output/'+save_dirname+'/model/netG/netG_epoch_{}.pth'.format(epoch))
        torch.save(netD.state_dict(), './output/'+save_dirname+'/model/netD/netD_epoch_{}.pth'.format(epoch))
        torch.save(optimizerG.state_dict(), './output/'+save_dirname+'/model/optimizerG/optimizerG_epoch_{}.pth'.format(epoch))
        torch.save(optimizerD.state_dict(), './output/'+save_dirname+'/model/optimizerD/optimizerD_epoch_{}.pth'.format(epoch))

    print('    total time: {} min , time for this epoch: {} s'
            .format(round(datetime.timedelta(seconds=time.time() - start_time).total_seconds()/60),
                    round(datetime.timedelta(seconds=time.time() - start_epoch_time).total_seconds(), 2)))

# Save the final trained model
torch.save(netG.state_dict(), './output/'+save_dirname+'/model/netG/netG_final.pth')
torch.save(netD.state_dict(), './output/'+save_dirname+'/model/netD/netD_final.pth')
torch.save(optimizerG.state_dict(), './output/'+save_dirname+'/model/optimizerG/optimizerG_final.pth')
torch.save(optimizerD.state_dict(), './output/'+save_dirname+'/model/optimizerD/optimizerD_final.pth')


fig, ax = plt.subplots(figsize=(7,5))
plt.plot(np.linspace(1, iters, iters), G_losses_iter,label="Generator", alpha=0.95)
plt.plot(np.linspace(1, iters, iters),D_losses_iter,label="Discriminator", alpha=0.95)
plt.legend(fontsize=14)
ax.set_xlabel('Iterration', size=15)
ax.set_ylabel('Loss', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('./output/'+save_dirname+'/loss/G_D_loss_iter.png')


fig, ax = plt.subplots(figsize=(7,5))
plt.plot(np.linspace(1, num_epochs, num_epochs), G_losses_epoch,label="Generator", alpha=0.95)
plt.plot(np.linspace(1, num_epochs, num_epochs),D_losses_epoch,label="Discriminator", alpha=0.95)
plt.legend(fontsize=14)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('./output/'+save_dirname+'/loss/G_D_loss_epoch.png')


fig, ax = plt.subplots(figsize=(7,5))
plt.plot(np.linspace(1, iters, iters), G_losses_iter,label="Generator", alpha=0.9)
plt.plot(np.linspace(1, iters, iters),D_losses_iter,label="Discriminator", alpha=0.9)
plt.legend(fontsize=14)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)
ax2 = ax.twiny()
x_epoch_label = np.linspace(0, num_epochs, 5, dtype=int).tolist()
x_epoch_label[0] = 1
ax2.set_xticks([len(train_dataloader)*e for e in x_epoch_label])
ax2.set_xticklabels(x_epoch_label)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('./output/'+save_dirname+'/loss/G_D_loss.png', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(7,5))
plt.plot(np.linspace(1, iters, iters), D_output_real,label=r'Real: $D(\mathbf{x})$', alpha=0.75)
plt.plot(np.linspace(1, iters, iters),D_output_fake,label=r'Fake: $D(G(\mathbf{z}))$', alpha=0.75)
plt.plot([1,iters],[0.5,0.5], c = 'gray', alpha = 0.9, zorder=1)
plt.legend(fontsize=14)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator output', size=15)
ax.set_ylim(0,1.0)
ax2 = ax.twiny()
x_epoch_label = np.linspace(0, num_epochs, 5, dtype=int).tolist()
x_epoch_label[0] = 1
ax2.set_xticks([len(train_dataloader)*e for e in x_epoch_label])
ax2.set_xticklabels(x_epoch_label)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epoch', size=15)
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('./output/'+save_dirname+'/loss/Discriminator_output.png', bbox_inches='tight')


# save training progression data to csv files
lossdata_iter = np.array([np.linspace(1, iters, iters, dtype=int).tolist(), G_losses_iter, D_losses_iter, D_output_real, D_output_fake]).T
lossdata_iter_header = 'iter,G_losses_iter,D_losses_iter,D_output_real,D_output_fake'
np.savetxt('./output/'+save_dirname+'/loss/lossdata_iter.csv', lossdata_iter, header= lossdata_iter_header,fmt = "%.18f", delimiter = ",", comments = "")

lossdata_epoch = np.array([np.linspace(1, num_epochs, num_epochs, dtype=int), G_losses_epoch, D_losses_epoch]).T
lossdata_epoch_header = 'epoch,G_losses_epoch,D_losses_epoch'
np.savetxt('./output/'+save_dirname+'/loss/lossdata_epoch.csv', lossdata_epoch, header= lossdata_epoch_header,fmt = "%.18f", delimiter = ",", comments = "")

noisedata = fixed_noise.cpu().numpy().copy()
np.savetxt('./output/'+save_dirname+'/training_progression/fixed_noise_BS{}_nz{}.csv'.format(batch_size,nz), noisedata, fmt = "%.18f", delimiter = ",", comments = "")