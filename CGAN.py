from re import L
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from torchsummary import summary
import random

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        # self.fc = nn.Sequential(
        #     nn.Linear(self.input_dim + self.class_num, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512 * 4 * 4),
        #     nn.BatchNorm1d(512 * 4 * 4),
        #     nn.ReLU(),
        # )


        # self.deconv = nn.ModuleList([])
        # for d in range(2, int(np.log2(self.input_size))):
        #     if d < 6:
        #         in_ch, out_ch = 512, 512
        #     else:
        #         in_ch, out_ch = int(512 / 2**(d - 6)), int(512 / 2**(d - 5))
            
        #     if d == (int(np.log2(self.input_size)) - 1):
        #         self.deconv.append(nn.Sequential(
        #             nn.ConvTranspose2d(in_ch, self.output_dim, 4, 2, 1),
        #             nn.ReLU(),
        #         ))
        #     else:
        #         self.deconv.append(nn.Sequential(
        #             nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
        #             nn.BatchNorm2d(out_ch),
        #             nn.ReLU(),
        #         ))

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )



        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4) , (self.input_size // 4))
        x = self.deconv(x)
        # for i, block in enumerate(self.deconv):
        #     x = block(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        # self.conv = nn.ModuleList([])
        # for d in reversed(range(2, int(np.log2(self.input_size)))): 
        #     if d == int(np.log2(self.input_size)) - 1:
        #         in_ch, out_ch = (self.input_dim + self.class_num), int(512 / 2**(d - 6))
        #     elif d < 6:
        #         in_ch, out_ch = 512, 512
        #     else:
        #         in_ch, out_ch = int(512 / 2**(d - 5)), int(512 / 2**(d - 6))
                
            
        #     if d == 2:
        #         self.conv.append(nn.Sequential(
        #             nn.Conv2d(in_ch, out_ch, 4, 2, 1),
        #             nn.LeakyReLU(0.2)
        #         ))
        #     else:
        #         self.conv.append(nn.Sequential(
        #             nn.Conv2d(in_ch, out_ch, 4, 2, 1),
        #             nn.BatchNorm2d(out_ch),
        #             nn.LeakyReLU(0.2)
        #         ))

        # self.fc = nn.Sequential(
        #     nn.Linear(512 * 4 * 4, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, self.output_dim),
        #     nn.Sigmoid(),
        # )




        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        # for i, block in enumerate(self.conv):
        #     x = block(x)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

class CGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.exp_name = args.exp[args.exp.find("_d")+1:]
        self.z_dim = 62
        
        self.data_loader = dataloader(args, self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()
        self.class_num = sum(self.data_loader.dataset.n_each_task) + len(self.data_loader.dataset.continuous_column)
        self.sample_num = 121

        # load dataset
        y_ = torch.concat([data[1], data[2]], axis=1)
        y_vec_ = y_
        y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
        print("y_vec_:")
        print(y_vec_[0, :])
        print("y_fill_:")
        print(y_fill_[0, :])


        

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data[0].shape[1], input_size=self.input_size, class_num=self.class_num)
        self.D = discriminator(input_dim=data[0].shape[1], output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # Generate sample
        if self.exp_name == "dDI_c":
            self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
            for i in range(self.class_num):
                self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
                for j in range(1, self.class_num):
                    self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

            temp = torch.zeros((self.class_num, 1))
            for i in range(self.class_num):
                temp[i, 0] = i

            temp_y = torch.zeros((self.sample_num, 1))
            for i in range(self.class_num):
                temp_y[i*self.class_num: (i+1)*self.class_num] = temp


            self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)


        elif self.exp_name == "dAR_HR_VR_DI_c":
            self.sample_z_ = torch.zeros((121, self.z_dim))
            for i in range(11):
                self.sample_z_[i*11] = torch.rand(1, self.z_dim)
                for j in range(1, 11):
                    self.sample_z_[i*11 + j] = self.sample_z_[i*11]

            self.sample_y_ = torch.zeros((121, 19)).type(torch.LongTensor)
            for i in range(11):
                 # C307 x2
                if i < 2:
                    temp = torch.tensor([0,1,0,0,1,1,0,0])
                # C315 x2
                elif 2 <= i < 4:
                    temp = torch.tensor([0,1,0,1,0,0,1,0])
                # C330 x2
                elif 4 <= i < 6:
                    temp = torch.tensor([0,1,0,1,0,0,0,1])
                # C615 x2
                elif 6 <= i < 8:
                    temp = torch.tensor([0,0,1,1,0,0,1,0])
                # C1050 x3
                else:
                    temp = torch.tensor([1,0,0,1,0,0,1,0])
                self.sample_y_[i * 11, :8]= temp
                for j in range(11):
                    self.sample_y_[i*11 + j] = self.sample_y_[i*11]
                    self.sample_y_[i*11 + j][8] = 0
                    self.sample_y_[i*11 + j][j+8] = 1

        elif self.exp_name == "dAR_HR_VR_cDI":
            self.sample_z_ = torch.zeros((121, self.z_dim))
            for i in range(11):
                self.sample_z_[i*11] = torch.rand(1, self.z_dim)
                for j in range(1, 11):
                    self.sample_z_[i*11 + j] = self.sample_z_[i*11]

            self.sample_y_ = torch.zeros((121, 9)).type(torch.float32)
            for i in range(11):
                 # C307 x2
                if i < 2:
                    temp = torch.tensor([0,1,0,0,1,1,0,0])
                # C315 x2
                elif 2 <= i < 4:
                    temp = torch.tensor([0,1,0,1,0,0,1,0])
                # C330 x2
                elif 4 <= i < 6:
                    temp = torch.tensor([0,1,0,1,0,0,0,1])
                # C615 x2
                elif 6 <= i < 8:
                    temp = torch.tensor([0,0,1,1,0,0,1,0])
                # C1050 x3
                else:
                    temp = torch.tensor([1,0,0,1,0,0,1,0])
                self.sample_y_[i * 11, :8]= temp
                for j in range(11):
                    self.sample_y_[i*11 + j] = self.sample_y_[i*11]
                    self.sample_y_[i*11 + j][8] = j/10.

        elif self.exp_name == "dDI_cAR_HR_VR":
            self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
            for i in range(11):
                self.sample_z_[i*11] = torch.rand(1, self.z_dim)
                for j in range(1, 11):
                    self.sample_z_[i*11 + j] = self.sample_z_[i*11]

            self.sample_y_ = torch.zeros((121, 14)).type(torch.float32)
            for i in range(11):
                 # C307 x2
                if i < 2:
                    temp = torch.tensor([3.0, 0.742, 1.282])
                # C315 x2
                elif 2 <= i < 4:
                    temp = torch.tensor([3.0, 1.444, 1.106])
                # C330 x2
                elif 4 <= i < 6:
                    temp = torch.tensor([3.0, 2.889, 1.106])
                # C615 x2
                elif 6 <= i < 8:
                    temp = torch.tensor([6.0, 1.444, 1.106])
                # C1050 x3
                else:
                    temp = torch.tensor([10.0, 1.444, 1.106])
                self.sample_y_[i * 11, -3:]= temp
                for j in range(11):
                    self.sample_y_[i*11 + j] = self.sample_y_[i*11]
                    self.sample_y_[i*11 + j][0] = 0
                    self.sample_y_[i*11 + j][j] = 1

        elif self.exp_name == "d_cAR_HR_VR_DI":
            self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
            for i in range(11):
                self.sample_z_[i*11] = torch.rand(1, self.z_dim)
                for j in range(1, 11):
                    self.sample_z_[i*11 + j] = self.sample_z_[i*11]

            self.sample_y_ = torch.zeros((121, 4)).type(torch.float32)
            for i in range(11):
                 # C307 x2
                if i < 2:
                    temp = torch.tensor([3.0, 0.742, 1.282])
                # C315 x2
                elif 2 <= i < 4:
                    temp = torch.tensor([3.0, 1.444, 1.106])
                # C330 x2
                elif 4 <= i < 6:
                    temp = torch.tensor([3.0, 2.889, 1.106])
                # C615 x2
                elif 6 <= i < 8:
                    temp = torch.tensor([6.0, 1.444, 1.106])
                # C1050 x3
                else:
                    temp = torch.tensor([10.0, 1.444, 1.106])
                self.sample_y_[i * 11, :3]= temp
                for j in range(11):
                    self.sample_y_[i*11 + j] = self.sample_y_[i*11]
                    self.sample_y_[i*11 + j][3] = j/10.

        print("First sample_y_:")
        print(self.sample_y_[:11])

        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, data in enumerate(self.data_loader):
                x_ = data[0]
                y_ = torch.concat([data[1], data[2]], axis=1)

                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = y_
                # y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.class_num, self.input_size, self.input_size)
                
                if self.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake = self.D(G_, y_fill_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if (epoch+1) % 20 == 0:
              with torch.no_grad():
                  self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        # utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.exp + '/' + self.model_name,
        #                          self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.result_dir, "model"), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(self.result_dir , "progress", self.model_name + '_epoch%05d' % (epoch) + '.png'))

    def save(self):
        save_dir = os.path.join(self.result_dir, "model")

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.result_dir, "model")

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

if __name__ == "__main__":
    G = generator(input_dim=62, output_dim=3, input_size=512, class_num=10)
    noise = torch.rand((2, 62))
    label = torch.rand((2, 10))
    out = G(noise, label)
    print(G)
    D = discriminator(input_dim=3, output_dim=1, input_size=512, class_num=10)
    label = torch.rand((2, 10, 512, 512))
    out = D(out, label)
    print(D)
