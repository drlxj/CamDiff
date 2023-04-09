# CamDiff

## Create environment
```` bash
# Create env
conda create -n camdiff python=3.10
conda activate camdiff

# Cuda 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

!pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate
pip install git+https://github.com/openai/CLIP.git
````

## Image Generation
Download COD datasets in the Dataset folder. The dataset needs to include 'Imgs' and 'GT' folder.
```` bash
python inpainting_diff.py --indir ./Dataset --outdir ./result
```` 

## Generation Results 
The figure shows the synthesized images from our framework from various classes. On the left side, the text is the input prompt, and each image is extended to generate three additional images of the same class, featuring objects with varying appearances. 
![Figure 1 - gneration](Imgs/multi.png)

## Evaluation on the COD models
![Figure 2 - gneration](Imgs/eval.png)

