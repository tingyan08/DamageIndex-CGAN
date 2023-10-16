import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import save_images


if __name__ == "__main__":
    plt.rcParams["savefig.bbox"] = 'tight'
    root = './Plane_data_750x800/images'
    exp = "C615"
    image_list = []
    for im in sorted(glob.glob(os.path.join(root, f"{exp}*E.JPG"))):
        image_list.append(np.expand_dims(np.asarray(Image.open(im).resize((128, 128))), axis = 0)/255)
    image_list = np.concatenate(image_list, axis=0)
    save_images(image_list, [1, 11], f'./real_images_visualize/real_{exp}_2.png')
    print("finished!")



















