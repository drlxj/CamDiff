import PIL
import os
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

root = "COD10K-CAM-3-Flying-53-Bird-3016.jpg"
test3 = os.path.join("/cluster/work/cvl/denfan/Train/out/test3", root)
test4 = os.path.join("/cluster/work/cvl/denfan/Train/out/test4", root)
num_samples = 5
images = []
images_no = []
for i in range(num_samples):
    img = Image.open(os.path.splitext(test3)[0] + "-" + str(i+1) + os.path.splitext(test3)[1])
    img_no = Image.open(os.path.splitext(test4)[0] + "-" + str(i+1) + os.path.splitext(test4)[1])
    images.append(img)
    images_no.append(img_no)

grid = image_grid(images, 1, num_samples)
grid_no = image_grid(images_no, 1, num_samples)
grid.save("./result/grid_bird.png")
grid_no.save("./result/grid_no_bird.png")