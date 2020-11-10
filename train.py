from __future__ import print_function
import argparse
import os
import numpy as np
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import autograd
from torchvision.utils import save_image

from datasets import ImageDataset
from model import define_D, define_G, get_scheduler, GANLoss, update_learning_rate

cudnn.benchmark = True


def calculate_gradient_penalty(disc, input, real_images, fake_images, device):
    eta = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0,1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device)
    #interpolated = torch.cat((input, interpolated), 1)
   
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = disc(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def save_sample(net_g, batches_done, testing_data_loader, dataset_dir, result_folder, device):
    sample = next(iter(testing_data_loader))
    samples = sample[1].to(device)
    masked_samples = sample[0].to(device)
    # mask = sample[2].to(device)

    # Generate inpainted image
    output = net_g(masked_samples)
    gen_masks = torch.max(output, 1, keepdim=True)[1].float()
    # filled_samples = gen_masks
    filled_samples = output
    
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -1)
    save_image(filled_samples.data, dataset_dir + ('%d.png' % batches_done), normalize=True)
    save_image(sample, result_folder + ('%d.png' % batches_done), nrow=1, normalize=True)


def train(img_size=64, channels=1, num_classes=3, batch_size=32,
          dataset_dir='./size_64/20_den/', result_folder='./size_64/Wpix2pix_ppath/',
          epoch_count=1, niter=100, niter_decay=100, lr_decay_iters=50):

    os.makedirs(result_folder, exist_ok=True)

    # Dataset loader
    training_data_loader = DataLoader(ImageDataset(dataset_dir, img_size=img_size),
                                      batch_size=batch_size, shuffle=True)
    testing_data_loader = DataLoader(ImageDataset(dataset_dir, mode='val', img_size=img_size),
                                     batch_size=6, shuffle=True, num_workers=1)

    gpu_id = 'cuda:3'
    device = torch.device(gpu_id)

    print('===> Building models')
    net_g = define_G(channels, num_classes, 64, 'batch', False, 'normal', 0.02,
                     gpu_id=gpu_id, use_ce=False, use_attn=True, context_encoder=False, unet=False)

    net_d = define_D(channels, 64, 'basic', gpu_id=gpu_id)

    weight = torch.FloatTensor([1, 1, 1]).to(device)

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)
    criterionCE = nn.CrossEntropyLoss(weight=weight).to(device)

    lr = 0.0002
    beta1 = 0.5
    lr_policy = 'lambda'

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, lr_policy)
    net_d_scheduler = get_scheduler(optimizer_d, lr_policy)

    loss_history = {'G': [], 'D': [], 'p': [], 'adv': [], 'valPSNR': []}

    for epoch in range(epoch_count, niter + niter_decay + 1):
        # train
        for iteration, batch in enumerate(training_data_loader, 1):
            # forward
            real_a, real_b, path = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # imshow(torch.cat((real_a[0], real_b[0]), -1).cpu().detach().numpy().reshape(img_size, img_size * 2))
            # imshow(real_b[0].cpu().detach().numpy().reshape(img_size, img_size))

            output = net_g(real_a)

            fake_b = output
            # fake_b = torch.max(output, 1, keepdim=True)[1].float()
            # fake_path = torch.where(fake_b == 0, torch.ones_like(fake_b).to(device),
            #                         torch.zeros_like(fake_b).to(device)).to(device)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()

            # train with fake
            # fake_ab = torch.cat((real_a, fake_b), 1)
            # fake_ab = torch.cat((real_a, fake_path), 1)
            # pred_fake = net_d.forward(fake_ab.detach())
            pred_fake = net_d.forward(fake_b.detach())

            # pred_fake = net_d.forward(fake_path.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            # eal_ab = torch.cat((real_a, real_b), 1)
            # real_ab = torch.cat((real_a, path), 1)
            # pred_real = net_d.forward(real_ab)
            pred_real = net_d.forward(real_b)

            # pred_real = net_d.forward(path)
            loss_d_real = criterionGAN(pred_real, True)

            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()

            gradient_penalty = calculate_gradient_penalty(net_d, real_a.data, real_b.data, fake_b.data, device)
            # gradient_penalty = calculate_gradient_penalty(net_d, real_a.data, path.data, fake_path.data, device)
            gradient_penalty.backward()

            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            # fake_ab = torch.cat((real_a, fake_b), 1)
            # fake_ab = torch.cat((real_a, fake_path), 1)
            # pred_fake = net_d.forward(fake_ab)
            pred_fake = net_d.forward(fake_b)

            # pred_fake = net_d.forward(fake_path)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * 10
            # loss_g_ce = criterionCE(output, real_b[:, 0, ...].long()) * 10
            # loss_len = (torch.mean(path) - torch.mean(fake_path)).pow(2)
            loss_g = loss_g_gan + loss_g_l1  # + loss_len

            loss_g.backward()

            optimizer_g.step()


            # print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(epoch,
            #                                                                     iteration,
            #                                                                     len(training_data_loader),
            #                                                                     loss_d.item(),
            #                                                                     loss_g.item()))

            loss_history['D'].append(loss_d.item())
            loss_history['G'].append(loss_g.item())
            loss_history['p'].append(loss_g_l1.item())

            # if iteration % 50 == 0:
            #     clear_output(True)
            #     plt.figure(figsize=[6, 4])
            #     plt.title("G vs D losses over time")
            #     plt.plot(loss_history['D'], linewidth=2, label='Discriminator')
            #     plt.plot(loss_history['G'], linewidth=2, label='Generator')
            #     plt.legend()
            #     plt.show()

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

        # test
        avg_psnr = 0
        for batch in testing_data_loader:
            input, target, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            output = net_g(input)
            prediction = output
            # prediction = torch.max(output, 1, keepdim=True)[1].float()
            mse = criterionMSE(prediction, target)
            psnr = 10 * log10(1 / (mse.item() + 1e-16))
            avg_psnr += psnr

        loss_history['valPSNR'] += [avg_psnr / len(testing_data_loader)]
        # print(len(testing_data_loader))
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

        #checkpoint
        save_sample(net_g,
                    epoch * len(training_data_loader) + iteration,
                    testing_data_loader,
                    dataset_dir,
                    result_folder,
                    device)

        torch.save(net_g.state_dict(), result_folder + 'generator.pt')
        torch.save(net_d.state_dict(), result_folder + 'discriminator.pt')
        np.save(result_folder + 'loss_history.npy', loss_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input/output grid.')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels in the input image.')
    parser.add_argument('--num_classes', type=int, default=3, help='Output number of channels/classes.')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='Path to the dataset with images.')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Where all the results/weights will be saved.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='From which epoch to start.')
    parser.add_argument('--number_of_epochs', type=int, default=100,
                        help='Number of epochs to train.')

    parsed_args = parser.parse_args()
    train(parsed_args.img_size, parsed_args.channels, parsed_args.num_classes,
          parsed_args.batch_size, parsed_args.dataset_dir, parsed_args.results_dir,
          parsed_args.epoch_count, parsed_args.number_of_epochs)
