import argparse, os, torch
from CGAN import CGAN
from CGAN_plus import CGAN_plus
from fid_score import calculate_fid_given_paths
from samples_one_image import generate_sample, save

import json


# python main.py --gan_type CGAN --dataset damage-index --epoch 2000 --batch_size 8 --input_size 128 --discrete_column DI

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CGAN_plus',
                        choices=[ 'CGAN', 'CGAN_plus', 'infoGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='damage-index', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed', 'damage-index'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=128, help='The size of input image')
    parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "HR", "VR", "DI"], help='The discrete label of input')
    parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--exp', type=str, default='', help='Name of experiment')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    discrete = ""
    for i, dc in enumerate(args.discrete_column):
        if i == 0:
            discrete += dc
        else:
            discrete += f"_{dc}"

    continuous = ""
    for i, cc in enumerate(args.continuous_column):
        if i == 0:
            continuous += cc
        else:
            continuous += f"_{cc}"
    args.exp = f"{args.gan_type}_s{args.input_size}x{args.input_size}_b{args.batch_size}_e{args.epoch}_lrG{args.lrG}_lrD{args.lrD}_d{discrete}_c{continuous}"
    
    # --result_dir
    args.result_dir = os.path.join(args.result_dir, args.dataset, args.gan_type, args.exp)
    if not os.path.exists(os.path.join(args.result_dir, "progress")):
        os.makedirs(os.path.join(args.result_dir, "progress"))

    if not os.path.exists(os.path.join(args.result_dir, "model")):
        os.makedirs(os.path.join(args.result_dir, "model"))

    

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'CGAN_plus':
        gan = CGAN_plus(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

        # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

    # Generate Sample
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    fake_path = f"{args.result_dir}/fake_image"
    label_name = args.exp[args.exp.find("_d")+1:]
    sample_z_, sample_y_ = generate_sample(label_name, sample_num=110, z_dim=62, device=device)   
    with torch.no_grad():
        gan.G.eval()
        output = gan.G(sample_z_, sample_y_)
        output = (output + 1) / 2

    if not os.path.isdir(fake_path):
        os.makedirs(fake_path)
    save(output.cpu().data.numpy().transpose(0, 2, 3, 1), fake_path)
    print(f" [*] Sampling finished! Save at {fake_path} ! ")


    # Calculate FID
    real_path = "./Plane_data_750x800/images"
    batch_size = 64
    target_dim = 2048
    num_workers = min(os.cpu_count() , 8)
    fid_value = calculate_fid_given_paths(real_path,
                                          fake_path,
                                          batch_size,
                                          device,
                                          target_dim,
                                          num_workers)
    print(' [*] FID: ', fid_value)

if __name__ == '__main__':
    main()
