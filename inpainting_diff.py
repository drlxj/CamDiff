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
RATIO = 0.0625
RATIO_MIN = 0.0625
RATIO_MAX = 0.25
LENGTH_RATIO_MIN = 1/5
LENGTH_RATIO_MAX = 5
MASK_RATIO = 0.75
SHRINK = np.sqrt(MASK_RATIO)
PROB = 0.4
MASK_WIDTH = 128
MASK_HEIGHT = 128


def make_mask(mask, image):
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    image = np.array(image.convert("RGB"))
    image = image.transpose(2,0,1)
    # increase mask to box
    coord  = np.where(mask == 1)
    xmin = min(coord[0])
    xmax = max(coord[0])
    ymin = min(coord[1])
    ymax = max(coord[1])

    new_image, new_mask, mask_ratio, coord, flag = choose_area(xmin, xmax, ymin, ymax, image)

    if flag == 1:
        new_image = Image.fromarray(new_image.astype(np.uint8).transpose(1, 2, 0))
        mask_image = Image.fromarray(new_mask.astype(np.uint8)*255).convert("RGB") 
    else:
        mask_image = 0 
 
    return new_image, mask_image, mask_ratio, coord, flag

def choose_area(xmin, xmax, ymin, ymax, image):
    A = np.array([[0, 0],         [xmin, ymin]])
    B = np.array([[0, ymin],      [xmin, ymax]])
    C = np.array([[0, ymax],      [xmin, WIDTH]])
    D = np.array([[xmin, 0],      [xmax, ymin]])
    E = np.array([[xmin, ymax],   [xmax, WIDTH]])
    F = np.array([[xmax, 0],      [HEIGHT, ymin]])
    G = np.array([[xmax, ymin],   [HEIGHT, ymax]])
    H = np.array([[xmax, ymax], [HEIGHT, WIDTH]])

    candidates = [A, B, C, D, E, F, G, H]
    random.shuffle(candidates)
    flag = 0
    for i in candidates:    
        mask_ratio = (i[1, 0] - i[0, 0]) * (i[1, 1] - i[0, 1]) / (WIDTH * HEIGHT) 
        if mask_ratio > RATIO_MIN:                              # avoid mask ratio is zero
            # Mask is a square, because DM's input size is 512 x 512
            if ((i[1, 0] - i[0, 0]) < (i[1, 1] - i[0, 1])):
                i[1, 1] = i[0, 1] + (i[1, 0] - i[0, 0])
            else:
                i[1, 0] = i[0, 0] + (i[1, 1] - i[0, 1])
            if mask_ratio > RATIO_MAX:                          # avoid mask ratio is too big
                shrink = np.sqrt(RATIO_MAX / mask_ratio)
                x_mid = int((i[1, 0] + i[0, 0]) / 2)
                y_mid = int((i[1, 1] + i[0, 1]) / 2)
                dx = int((i[1, 0] - i[0, 0]) * shrink)
                dy = int((i[1, 1] - i[0, 1]) * shrink)
                d = min(dx, dy)
                i[0, 0] = int(x_mid - dx / 2)
                i[1, 0] = int(x_mid + dx / 2)
                i[0, 1] = int(y_mid - dy / 2)
                i[1, 1] = int(y_mid + dy / 2)
            # new_mask[i[0, 0]:i[1, 0], i[0, 1]:i[1, 1]] = 1
            new_image = image[:, i[0, 0]:i[1, 0], i[0, 1]:i[1, 1]]
            flag += 1
            break   
    if flag == 1:
        new_mask = np.zeros((new_image.shape[1], new_image.shape[2]))
        x_mid_mask = int(new_image.shape[1] / 2)
        y_mid_mask = int(new_image.shape[2] / 2)
        dx_half_mask = int(new_image.shape[1] * SHRINK  / 2)
        dy_half_mask = int(new_image.shape[2] * SHRINK / 2)
        new_mask[(x_mid_mask-dx_half_mask) : (x_mid_mask+dx_half_mask), (y_mid_mask-dy_half_mask):(y_mid_mask+dy_half_mask)] = 1

        mask_ratio = (i[1, 0] - i[0, 0]) * (i[1, 1] - i[0, 1]) / (WIDTH * HEIGHT) * MASK_RATIO
    else:
        new_mask = 0
        new_image = 0
        mask_ratio = 0
        i = 0

    return new_image, new_mask, mask_ratio, i, flag


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

def num_bad_img(images, mask_image, prompt, org_w , org_h, coord, org_image):
    del_idx = []
    left_images = []
    for idx, image in enumerate(images):
        test_object = crop_object(image, mask_image)
        label, prob = classifier.forward(test_object)

        # avoid many types of fish
        if "Fish" in label or "fish" in label:
            label = "Fish"
        if "Frogmouth" in label:
            label = "Bird"

        # insert the sampled image into the original image
        image = image.resize((org_w, org_h))
        image = np.array(image.convert("RGB"))
        image = image.transpose(2,0,1)

        new_image = org_image.copy()
        new_image = np.array(new_image.convert("RGB"))
        new_image = new_image.transpose(2,0,1)
        new_image[:, coord[0, 0]:coord[1, 0], coord[0,1]:coord[1,1]] = image
        new_image = Image.fromarray(new_image.transpose(1, 2, 0))
        # new_image.save("./image.jpg")
        # breakpoint()

        if label not in prompt or prob < PROB:
            del_idx.append(idx)
        else:
            left_images.append(new_image)
    
    return len(del_idx), left_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default="./Dataset",
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./result",
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

    data_root = os.path.join(opt.indir, "Imgs")
    mask_root = os.path.join(opt.indir, "GT")

    images =  [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
    masks = [os.path.join(mask_root, os.path.splitext(os.path.split(file_path)[-1])[0] + '.png') for file_path in images]
    print(f"Found {len(masks)} inputs.")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(opt.device)
    print("Pretrained model is loaded")
    classifier = ClipPipeline(data_root, opt.device)

    print("-------------Begin inpainting-------------")
    start = time.time()
    os.makedirs(opt.outdir, exist_ok=True)
    for image_path, mask_path in zip(images, masks): 
        print(f"Image file: {image_path}")
        # breakpoint()
        outpath = os.path.join(opt.outdir, os.path.split(image_path)[1])
        if len(os.path.split(outpath)[1].split("-")) == 1:
            # camo, chameleon, nc4k
            prompt = "a " + random.choice(classifier.labels)
        else:
            prompt = "a " + os.path.split(outpath)[1].split("-")[-2]        
        print("Prompt: " + prompt) 
        # avoid many types of fish
        if "Fish" in prompt or "fish" in prompt:
            prompt = "a Fish"
        if "Frogmouth" in prompt:
            prompt = "a Bird"

        #image and mask_image should be PIL images.
        #The mask structure is white for inpainting and black for keeping as is
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = image.resize((WIDTH, HEIGHT))
        mask= mask.resize((WIDTH, HEIGHT))
        print(f"resized to ({WIDTH}, {HEIGHT})")    

        # Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
        # usually at the expense of lower image quality.
        num_samples = 1
        guidance_scale= 7.5
        seed = 0
        for i in range(num_samples): 
            if len(os.path.split(outpath)[1].split("-")) == 1:
                # camo, chameleon, nc4k
                prompt = "a " + random.choice(classifier.labels)
            seed = random.randint(seed + 1,  seed + 10)
            # mask position is randomly generated
            new_image, mask_image, mask_ratio, coord, flag = make_mask(mask, image)
            print(f"mask ratio is {mask_ratio}")
            
            if flag == 0:
                print("Remask")
                continue

            org_w , org_h = mask_image.size
            new_image = new_image.resize((WIDTH, HEIGHT))
            mask_image= mask_image.resize((WIDTH, HEIGHT))

            generator = torch.Generator(device="cuda").manual_seed(seed) # change the seed to get different results
            
            images = pipe(prompt=prompt, 
                        image=new_image, 
                        mask_image=mask_image,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        num_images_per_prompt=1,
                        ).images
            num_resamples, images = num_bad_img(images, mask_image, prompt, org_w , org_h, coord, image)
            
            # avoid no break in while loop
            count = 0
            while (len(images) < 1) & (count < 10):
                print(f"Resample {num_resamples} images")
                new_image, mask_image, mask_ratio, coord, flag = make_mask(mask, image)
                print(f"mask ratio is {mask_ratio}")

                if flag == 0:
                    print("Remask")
                    continue

                org_w , org_h = mask_image.size
                new_image = new_image.resize((WIDTH, HEIGHT))
                mask_image= mask_image.resize((WIDTH, HEIGHT))

                generator = torch.Generator(device="cuda").manual_seed(random.randint(seed + 1,  seed + 10))
                resample_images = pipe(prompt=prompt, 
                        image=new_image, 
                        mask_image=mask_image,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        num_images_per_prompt=num_resamples,
                        ).images

                num_resamples, left_images = num_bad_img(resample_images, mask_image, prompt, org_w , org_h, coord, image)
                for img in left_images:
                    images.append(img)
                count += 1
            
            if num_resamples != 1:
                subpath = os.path.join(os.path.splitext(outpath)[0] + "-" + str(i) + os.path.splitext(outpath)[1])
                images[0].save(subpath)

    end = time.time()
    print(f"Total time: {end - start}")
