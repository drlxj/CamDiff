# CamDiff
**CamDiff** is an approach inspired by AI-Generated Content (AIGC) that overcomes the scarcity of multi-pattern training images. Specifically, CamDiff leverages the [latent diffusion model](https://huggingface.co/runwayml/stable-diffusion-inpainting) to synthesize salient objects in camouflaged scenes, and use the zero-shot image classification ability of the [CLIP model](https://openai.com/research/clip) to prevent synthesis failures and ensure the synthesized object aligns with the input prompt. 

Consequently, the synthesized image retains its original camouflage label while incorporating salient objects, yielding camouflage samples with richer characteristics. The results of user studies show that the salient objects in the scenes synthesized by our framework attract the user's attention more; thus, such samples pose a greater challenge to existing COD models.

![Figure 1 - examples](Imgs/examp.png)
The synthesized object is within a green box, while the original object within the image was enclosed in a red box. 

## Requirement
A suitable [conda](https://conda.io/) environment named `camdiff` can be created and activated with:
```` bash
# Create env
conda create -n camdiff python=3.10
conda activate camdiff

# Cuda 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

!pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate
pip install git+https://github.com/openai/CLIP.git
````

## Usage
### Generation
Download COD datasets in the Dataset folder. The dataset needs to include 'Imgs' and 'GT' folder.
```` bash
python inpainting_diff.py --indir ./Dataset --outdir ./result
```` 
### Dataset
The generated dataset can be downloaded: 

## Paper Details
### Generation Results 
Examples of the synthesized images from CamDiff from various classes. Each image is extended to generate three additional images of the same class, featuring objects with varying appearances. 
![Figure 1 - gneration](Imgs/multi2.png)


### Evaluation on the robustness to saliency of COD models
The first row: the results of the models that were pretrained on COD datasets, and then tested on the original images. They show a good performance of detecting the camouflaged objects.

The second row: the results of the models that use the same checkpoints as the first row, and then tested on the synthesized images. Most of them detect the salient objects, which is undesired. Meanwhile, the accuracy of detecting camouflaged objects decrease. For example, SINet lose some parts compared with the mask in the first row. ZoomNet even ignores the camouflaged objects. The results demonstrate the COD methods lack robustness to saliency.

The third row: the results of the models that use the checkpoints trained on our Diff-COD dataset, and then tested on the synthesized images. Compared with the second row, the robustness to saliency improves significantly.

![Figure 2 - gneration](Imgs/Screenshot.png)

