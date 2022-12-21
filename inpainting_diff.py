from diffusers import StableDiffusionInpaintPipeline
import torch
import os
# from einops import repeat
import numpy as np
import time
import argparse
from PIL import Image
import random

# from efficientnet_classification import EfficientnetPipeline
from clip_classification import ClipPipeline

WIDTH = 512
HEIGHT = 512
RATIO = 0.05
PROB = 0.6

def make_mask(mask):
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # increase mask to box
    coord  = np.where(mask == 1)
    xmin = min(coord[0])
    xmax = max(coord[0])
    ymin = min(coord[1])
    ymax = max(coord[1])
    # expand the mask
    mask_ratio = (xmax-xmin) * (ymax-ymin) / (WIDTH * HEIGHT)
    if mask_ratio < RATIO:
        expand = np.sqrt(RATIO / mask_ratio)
        xmax = int(xmax*expand)
        ymax = int(ymax*expand)
        if xmax > WIDTH:
            xmax = WIDTH
        if ymax > HEIGHT:
            ymax = HEIGHT
    mask[xmin:(xmax+1), ymin:(ymax+1)] = 1
    
    mask_image = Image.fromarray(mask.astype(np.uint8)*255).convert("RGB")
 
    return mask_image, mask_ratio

def crop_object(image, mask):
    image = np.array(image.convert("RGB"))
    image = image.transpose(2,0,1)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # get box info
    coord  = np.where(mask == 1)
    xmin = min(coord[0])
    xmax = max(coord[0])
    ymin = min(coord[1])
    ymax = max(coord[1])

    # dimension = RGB image
    mask = mask[None]  

    mask_image = image * (mask > 0.5)
    mask_image = Image.fromarray(mask_image[:, xmin:xmax, ymin:ymax].transpose(1, 2, 0))
    ## Save mask
    # mask_image = image * (mask < 0.5)
    # mask_image = Image.fromarray(mask_image.transpose(1, 2, 0))

    return mask_image

def num_bad_img(images):
    del_idx = []
    left_images = []
    for idx, image in enumerate(images):
        test_object = crop_object(image, mask_image)
        # test_object.save("mm.jpg")
        label, prob = classifier.forward(test_object, data_root)
        if label not in prompt or prob < PROB:
            del_idx.append(idx)
        else:
            left_images.append(image)
    
    return len(del_idx), left_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="/cluster/work/cvl/denfan/Train",
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/cluster/work/cvl/denfan/Train/out/test1",
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="computation device to use",
        choices=["cpu", "cuda"]
    )
    opt = parser.parse_args()

    data_root = os.path.join(opt.indir, "Image")
    mask_root = os.path.join(opt.indir, "GT_Object")

    # 2150:2300
    images = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]#[os.path.join(data_root, "COD10K-CAM-3-Flying-55-Butterfly-3420.jpg")]
    masks = [os.path.join(mask_root, os.path.splitext(os.path.split(file_path)[-1])[0] + '.png') for file_path in images]
    print(f"Found {len(masks)} inputs.")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(opt.device)
    print("Pretrained model is loaded")
    # classifier = EfficientnetPipeline(opt.device)
    classifier = ClipPipeline(opt.device)

    print("-------------Begin inpainting-------------")
    start = time.time()
    os.makedirs(opt.outdir, exist_ok=True)
    for image_path, mask_path in zip(images, masks): 
        print(f"Image file: {image_path}")
        outpath = os.path.join(opt.outdir, os.path.split(image_path)[1])
        prompt = "a " + os.path.split(outpath)[1].split("-")[-2]
        print("Prompt: " + prompt)

        #image and mask_image should be PIL images.
        #The mask structure is white for inpainting and black for keeping as is
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = image.resize((WIDTH, HEIGHT))
        mask= mask.resize((WIDTH, HEIGHT))
        # print(f"resized to ({WIDTH}, {HEIGHT})")
        # os.makedirs("/cluster/work/cvl/denfan/Train/metric_set", exist_ok=True)
        # image.save(os.path.join("/cluster/work/cvl/denfan/Train/metric_set/", os.path.split(image_path)[1]))

        mask_image, mask_ratio = make_mask(mask)
        print(f"mask ratio is {mask_ratio}")
        # mask_image.save("./m.jpg")
        # breakpoint()

        # Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
        # usually at the expense of lower image quality.
        num_samples = 10
        guidance_scale=7.5
        seed = random.randint(1, 10)
        generator = torch.Generator(device="cuda").manual_seed(seed) # change the seed to get different results
        
        images = pipe(prompt=prompt, 
                    image=image, 
                    mask_image=mask_image,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_samples,
                    ).images
        
        num_resamples, images = num_bad_img(images)
        count = 0
        while (len(images) < num_samples) & (count < 10):
            print(f"Resample {num_resamples} images")
            generator = torch.Generator(device="cuda").manual_seed(random.randint(1+seed, 1000+seed))
            resample_images = pipe(prompt=prompt, 
                    image=image, 
                    mask_image=mask_image,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_resamples,
                    ).images
            num_resamples, left_images = num_bad_img(resample_images)
            for img in left_images:
                images.append(img)
            count += 1

        for idx, image in enumerate(images, start = 1):
            # breakpoint()
            subpath = os.path.join(os.path.splitext(outpath)[0] + "-" + str(idx) + os.path.splitext(outpath)[1])
            image.save(subpath)

    end = time.time()
    print(f"Total time: {end - start}")