import os
import cv2
import json
import torch
import numpy as np
import utils
from argparse import Namespace



from CGAN import CGAN
from CGAN_plus import CGAN_plus


def generate_sample(label_name, sample_num=110, z_dim=62, device="cpu", exp="C307"):
    if label_name == "dDI_c":
        sample_z_ = torch.zeros((sample_num, z_dim))
        sample_z_[0] = torch.rand(1, z_dim)
        for j in range(1, 11):
            sample_z_[j] = sample_z_[0]

        temp = torch.zeros((11, 1))
        for i in range(11):
            temp[i, 0] = i

        sample_y_ = torch.zeros((sample_num, 11)).scatter_(1, temp.type(torch.LongTensor), 1)


    elif label_name == "dAR_HR_VR_DI_c":
        sample_z_ = torch.zeros((sample_num, z_dim))
        sample_z_[0] = torch.rand(1, z_dim)
        for j in range(1, 11):
            sample_z_[j] = sample_z_[0]

        sample_y_ = torch.zeros((sample_num, 19)).type(torch.LongTensor)
        # C307 x2
        if exp == "C307":
            temp = torch.tensor([0,1,0,0,1,1,0,0])
        # C315 x2
        elif exp == "C315":
            temp = torch.tensor([0,1,0,1,0,0,1,0])
        # C330 x2
        elif exp == "C330":
            temp = torch.tensor([0,1,0,1,0,0,0,1])
        # C615 x2
        elif exp == "C615":
            temp = torch.tensor([0,0,1,1,0,0,1,0])
        # C1050 x2
        elif exp == "C1015":
            temp = torch.tensor([1,0,0,1,0,0,1,0])

        elif exp == "CTR1":
            temp = torch.tensor([0,1,0,0,1,0,0,1]) #2.889
            # temp = torch.tensor([0,1,0,1,0,0,0,1]) #1.444

        sample_y_[:, :8]= temp
        for j in range(11):
            sample_y_[j][8] = 0
            sample_y_[j][j+8] = 1

    elif label_name == "dAR_HR_VR_cDI":
        sample_z_ = torch.zeros((sample_num, z_dim))
        sample_z_[0] = torch.rand(1, z_dim)
        for j in range(1, sample_num):
            sample_z_[j] = sample_z_[0]

        sample_y_ = torch.zeros((sample_num, 9)).type(torch.float32)
            # C307 x2
        if exp == "C307":
            temp = torch.tensor([0,1,0,0,1,1,0,0])
        # C315 x2
        elif exp == "C315":
            temp = torch.tensor([0,1,0,1,0,0,1,0])
        # C330 x2
        elif exp == "C330":
            temp = torch.tensor([0,1,0,1,0,0,0,1])
        # C615 x2
        elif exp == "C615":
            temp = torch.tensor([0,0,1,1,0,0,1,0])
        # C1050 x2
        elif exp == "C1015":
            temp = torch.tensor([1,0,0,1,0,0,1,0])
            
        elif exp == "CTR1":
            temp = torch.tensor([0,1,0,0,1,0,0,1]) #2.889
            # temp = torch.tensor([0,1,0,1,0,0,0,1]) #1.444

        sample_y_[:, :8]= temp
        for j in range(11):
            sample_y_[j] = sample_y_[0]
            sample_y_[j][8] = j/10.

    
    sample_z_ = sample_z_.to(device)
    sample_y_ = sample_y_.to(device) 
    

    return sample_z_, sample_y_


def save(output, path, prefix):
  
    for i, img in enumerate(output):
        cv2.imwrite(os.path.join(path, f"{prefix}_{i+1:02d}.jpg"), np.uint8(img*255)[:,:,::-1])


if __name__ == "__main__":
    num_example = 20
    column_name = "C315"
    model_name = "CGAN_plus"
    exp_name = "dAR_HR_VR_cDI"
    dataset = "damage-index(even)"
    exp = f"{model_name}_s128x128_b8_e40000_lrG0.0002_lrD0.0002_{exp_name}"
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
    if not os.path.exists(os.path.join(result_path, "sample", column_name)):
        os.makedirs(os.path.join(result_path, "sample", column_name))

    for i in range(num_example):
        sample_z_, sample_y_ = generate_sample(label_name, sample_num=11, z_dim=62, device=device, exp=column_name)   
        with torch.no_grad():
            gan.G.eval()
            output = gan.G(sample_z_, sample_y_)
            output = (output + 1) / 2
            output = output.cpu().data.numpy().transpose(0, 2, 3, 1)
        if not os.path.isdir(os.path.join(result_path, "sample")):
            os.makedirs(os.path.join(result_path, "sample"))
        # save(output, os.path.join(result_path, "sample"), "C1015")
        utils.save_images(output, [1, 11],
                            os.path.join(result_path, "sample", column_name, f"{column_name}_{i+1:02d}.jpg"))
    
        
    
    


