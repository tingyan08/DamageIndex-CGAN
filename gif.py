import os
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

def generate_animation(path, save_name, fps=20):
    images = []
    

    fig = plt.figure()
    img_name = glob.glob(os.path.join(path, "*.jpg"))[0]
    image = mpimg.imread(img_name)
    for j in range(11):
        plt.xticks([])
        plt.yticks([])
        if (j%2 == 0):
            plt.title(f"Damage Index = {j/10.:.1f}")
        else:
            plt.title(f"Damage Index = {j/10.:.1f} (unseen)")
        plt.imshow(image[:,j*128:(j+1)*128, :])
        plt.savefig(img_name.replace('.jpg', f'_temp{j:02d}.jpg'))
        images.append(img_name.replace('.jpg', f'_temp{j:02d}.jpg'))

    with imageio.get_writer(f'./gif/{save_name}.gif', mode='I', fps=fps) as writer:
        for filename in tqdm(images):
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)




if __name__ == "__main__":
    dataset = 'damage-index(even)'
    model = 'CGAN_plus' #CGAN CGAN_plus
    exp = 'dAR_HR_VR_cDI' # dDI_c dAR_HR_VR_DI_c dAR_HR_VR_cDI
    root = f'./results/{dataset}/{model}/{model}_s128x128_b8_e40000_lrG0.0002_lrD0.0002_{exp}/sample'

    print(f'{model}_{exp}')
    generate_animation(root + "/", f'{model}-{dataset}-{exp}-C1015', fps=2)