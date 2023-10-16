import os
import glob
from torchvision import transforms
from PIL import Image


path = "./Plane_data_750x800/images"
resize = transforms.Resize((128,128))


for i in glob.glob(os.path.join(path, "*.JPG")):
    img = Image.open(i)
    img = resize(img)
    img.save(i.replace("images", "resized"))

