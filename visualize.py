import os
import cv2
import json
import torch
import random
import numpy as np
from types import SimpleNamespace
from argparse import Namespace



from CGAN import CGAN
from CGAN_plus import CGAN_plus
from utils import save_images



model_name = "CGAN_plus"
dataset = "damage-index"
exp = f"{model_name}_s128x128_b8_e2000_lrG0.0002_lrD0.0002_dAR_HR_VR_cDI"
seed = 100
image_name = f"five_exp.jpg"
path = os.path.join("./results", dataset, model_name, exp)



torch.manual_seed(seed)

with open(os.path.join(path,"model", "config.json"), "r") as f:
    args = json.load(f)
    args = Namespace(**args)

args.gpu_mode = torch.cuda.is_available()
device = "cuda:0" if args.gpu_mode else "cpu"


# prepare input

# # # Experiment 1
# # Same Noise different DI
# condition = torch.zeros((11,11))
# for i in range(11):
#     condition[i][i] = 1
# condition = condition.to(device)
# z = torch.rand((1, 62)).to(device)
# z = z.repeat(11, 1)

# # Same DI different noise
# condition = torch.zeros((11,11))
# for i in range(11):
#     condition[i][10] = 1
# condition = condition.to(device)
# z = torch.rand((11, 62)).to(device)

# # Experiment 2 (AR (3) + HR (2) + VR (3) + DI (11))
# Same Noise / AR / HR / VR , different DI
# di = np.arange(0,1.1,0.1)
# condition = torch.zeros((len(di)*5,19))
# # C307 [0,1,0,0,1,1,0,0]
# # C315 [0,1,0,1,0,0,1,0]
# # C330 [0,1,0,1,0,0,0,1]
# # C615 [0,0,1,1,0,0,1,0]
# # C1015 [1,0,0,1,0,0,1,0]
# condition[0:11, 0:8] = torch.tensor([0,1,0,0,1,1,0,0])
# condition[11:22, 0:8] = torch.tensor([0,1,0,1,0,0,1,0])
# condition[22:33, 0:8] = torch.tensor([0,1,0,1,0,0,0,1])
# condition[33:44, 0:8] = torch.tensor([0,0,1,1,0,0,1,0])
# condition[44:55, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])

# for j in range(5):
#     for i, d in enumerate(di):
#         condition[j*11+i][8+i] = d
# condition = condition.to(device)
# z = torch.rand((1, 62)).to(device)
# z = z.repeat(len(di)*5, 1)

# # Same DI / AR / HR / VR,  different noise
# condition = torch.zeros((11,19))
# condition[:, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])
# for i in range(11):
#     condition[i][8] = 1
# condition = condition.to(device)
# z = torch.rand((11, 62)).to(device)

# # Same Noise / DI / HR / VR,  different AR
# condition = torch.zeros((3,19))
# condition[0, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])
# condition[1, 0:8] = torch.tensor([0,1,0,1,0,0,1,0])
# condition[2, 0:8] = torch.tensor([0,0,1,1,0,0,1,0])
# for i in range(3):
#     condition[i][13] = 1
# condition = condition.to(device)
# z = torch.rand((1, 62)).to(device)
# z = z.repeat(3, 1)

# # Same Noise / DI / AR / VR,  different HR
# condition = torch.zeros((2,19))
# condition[0, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])
# condition[1, 0:8] = torch.tensor([1,0,0,0,1,0,1,0])
# for i in range(2):
#     condition[i][16] = 1
# condition = condition.to(device)
# z = torch.rand((1, 62)).to(device)
# z = z.repeat(2, 1)

# # Same Noise / DI / HR / AR,  different VR
# condition = torch.zeros((3,19))
# condition[0, 0:8] = torch.tensor([1,0,0,1,0,1,0,0])
# condition[1, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])
# condition[2, 0:8] = torch.tensor([1,0,0,1,0,0,0,1])
# for i in range(2):
#     condition[i][16] = 1
# condition = condition.to(device)
# z = torch.rand((1, 62)).to(device)
# z = z.repeat(3, 1)

# # Experiment 3
# Same Noise / AR / HR / VR , different DI
di = np.arange(0,1.05,0.05)
condition = torch.zeros((len(di),9))
# C307 [0,1,0,0,1,1,0,0]
# C315 [0,1,0,1,0,0,1,0]
# C330 [0,1,0,1,0,0,0,1]
# C615 [0,0,1,1,0,0,1,0]
# C1015 [1,0,0,1,0,0,1,0]
condition[:, 0:8] = torch.tensor([0,0,1,1,0,0,1,0])
# condition[11:22, 0:8] = torch.tensor([0,1,0,1,0,0,1,0])
# condition[22:33, 0:8] = torch.tensor([0,1,0,1,0,0,0,1])
# condition[33:44, 0:8] = torch.tensor([0,0,1,1,0,0,1,0])
# condition[44:55, 0:8] = torch.tensor([1,0,0,1,0,0,1,0])

for j in range(1):
    for i, d in enumerate(di):
        condition[j*len(di)+i][8] = d
condition = condition.to(device)

z = torch.rand((1, 62)).repeat(len(di), 1).to(device)

# z = torch.rand((1, 62)).to(device)
# z = z.repeat(len(di)*5, 1)

if args.gan_type == 'CGAN':
    gan = CGAN(args)
elif args.gan_type == 'CGAN_plus':
    gan = CGAN_plus(args)
else:
    raise Exception("[!] There is no option for " + args.gan_type)

gan.load()
with torch.no_grad():
    gan.G.eval()
    output = gan.G(z, condition)
    output = (output + 1) / 2
    output = output.cpu().data.numpy().transpose(0, 2, 3, 1)
if not os.path.isdir(os.path.join("./unseen", exp)):
    os.makedirs(os.path.join("./unseen", exp))

for j in range(21):
    save_images(output[j:j+1, :, :, :], [1, 1], os.path.join("./unseen", exp, f"Noise_2_DI_{j/20.0:.2f}.png"))


# save_images(output, [5, 11], os.path.join("./visualization", exp, image_name))






