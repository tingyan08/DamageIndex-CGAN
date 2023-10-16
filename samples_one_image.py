import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import Namespace



from CGAN import CGAN
from CGAN_plus import CGAN_plus


def generate_sample(label_name, sample_num=110, z_dim=62, device="cpu"):
    num_row = int(sample_num/11)
    num_exp = int(num_row / 5)
    if label_name == "dDI_c":
        sample_z_ = torch.zeros((sample_num, z_dim))
        for i in range(num_row):
            sample_z_[i*11] = torch.rand(1, z_dim)
            for j in range(1, 11):
                sample_z_[i*11 + j] = sample_z_[i*11]

        temp = torch.zeros((11, 1))
        for i in range(11):
            temp[i, 0] = i

        temp_y = torch.zeros((sample_num, 1))
        for i in range(num_row):
            temp_y[i*11: (i+1)*11] = temp


        sample_y_ = torch.zeros((sample_num, 11)).scatter_(1, temp_y.type(torch.LongTensor), 1)


    elif label_name == "dAR_HR_VR_DI_c":
        sample_z_ = torch.zeros((sample_num, z_dim))
        for i in range(num_row):
            sample_z_[i*11] = torch.rand(1, z_dim)
            for j in range(1, 11):
                sample_z_[i*11 + j] = sample_z_[i*11]

        sample_y_ = torch.zeros((sample_num, 19)).type(torch.LongTensor)
        for i in range(num_row):
            # C307 
            if i < (1 * num_exp):
                temp = torch.tensor([0,1,0,0,1,1,0,0])
            # C315 
            elif (1 * num_exp) <= i < (2 * num_exp):
                temp = torch.tensor([0,1,0,1,0,0,1,0])
            # C330 
            elif (2 * num_exp) <= i < (3 * num_exp):
                temp = torch.tensor([0,1,0,1,0,0,0,1])
            # C615 
            elif (3 * num_exp) <= i < (4 * num_exp):
                temp = torch.tensor([0,0,1,1,0,0,1,0])
            # C1050 
            elif (4 * num_exp) <= i < (5 * num_exp):
                temp = torch.tensor([1,0,0,1,0,0,1,0])
            sample_y_[i * 11, :8]= temp
            for j in range(11):
                sample_y_[i*11 + j] = sample_y_[i*11]
                sample_y_[i*11 + j][8] = 0
                sample_y_[i*11 + j][j+8] = 1

    elif label_name == "dAR_HR_VR_cDI":
        sample_z_ = torch.zeros((sample_num, z_dim))
        for i in range(10):
            sample_z_[i*11] = torch.rand(1, z_dim)
            for j in range(1, 11):
                sample_z_[i*11 + j] = sample_z_[i*11]

        sample_y_ = torch.zeros((sample_num, 9)).type(torch.float32)
        for i in range(num_row):
            # C307 
            if i < (1 * num_exp):
                temp = torch.tensor([0,1,0,0,1,1,0,0])
            # C315 
            elif (1 * num_exp) <= i < (2 * num_exp):
                temp = torch.tensor([0,1,0,1,0,0,1,0])
            # C330 
            elif (2 * num_exp) <= i < (3 * num_exp):
                temp = torch.tensor([0,1,0,1,0,0,0,1])
            # C615 
            elif (3 * num_exp) <= i < (4 * num_exp):
                temp = torch.tensor([0,0,1,1,0,0,1,0])
            # C1050 
            elif (4 * num_exp) <= i < (5 * num_exp):
                temp = torch.tensor([1,0,0,1,0,0,1,0])
            sample_y_[i * 11, :8] = temp
            for j in range(11):
                sample_y_[i*11 + j] = sample_y_[i * 11]
                sample_y_[i*11 + j] = sample_y_[i*11]
                sample_y_[i*11 + j][8] = j/10.

    
    sample_z_ = sample_z_.to(device)
    sample_y_ = sample_y_.to(device) 
    

    return sample_z_, sample_y_


def save(output, path):
    for i, img in enumerate(output):
        cv2.imwrite(os.path.join(path, f"{i+1:05d}.jpg"), np.uint8(img*255)[:,:,::-1])


if __name__ == "__main__":
    model_name = "CGAN_plus"
    exp_name = "dAR_HR_VR_cDI"
    epoch = 40000
    dataset = "damage-index"
    exp = f"{model_name}_s128x128_b8_e{epoch}_lrG0.0002_lrD0.0002_{exp_name}"
    label_name = exp[exp.find("_d")+1:]
    result_path = f"./results/{dataset}/{model_name}/{exp}"

    with open(os.path.join(result_path, "model", "config.json"), "r") as f:
        args = json.load(f)
        args = Namespace(**args)

    args.gpu_mode = torch.cuda.is_available()
    device = "cuda:0" if args.gpu_mode else "cpu"

    if args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'CGAN_plus':
        gan = CGAN_plus(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)


    gan.load()


    results = []
    for i in tqdm(range(1)):
        sample_z_, sample_y_ = generate_sample(label_name, sample_num=110, z_dim=62, device=device)   
        with torch.no_grad():
            gan.G.eval()
            output = gan.G(sample_z_, sample_y_)
            output = (output + 1) / 2
            results.append(output)

    results = torch.cat(results, dim=0)
    results = results.cpu().data.numpy().transpose(0, 2, 3, 1)
    if not os.path.isdir(os.path.join(result_path, "fake_image")):
        os.makedirs(os.path.join(result_path, "fake_image"))
    save(results, os.path.join(result_path, "fake_image"))
        
    
    


