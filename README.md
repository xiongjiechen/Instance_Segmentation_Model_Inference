# Instance_Segmentation_Model_Inference
## Install PyTorch and MMDetection
Run the following command to create a conda environment for the installation of mmdetection:

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

Then install pytorch, mmengine, and mmcv.

On GPU platforms:
```
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
pip install mmpretrain
```

On CPU platforms:
```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
pip install -U openmim
pip install --upgrade pip setuptools wheel
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
pip install setuptools==60.2.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
pip install mmpretrain
```

Verify the installation:

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
You shoulbe be able to see the prediction results in the folder **./outputs** if mmdetection is successfully installed.

The official installation guide for mmdetection can be found here: https://mmdetection.readthedocs.io/en/latest/get_started.html

## Load instance segmentation model

Download inference scripts and model checkpoints via this link: https://drive.google.com/file/d/1pT3IERj6S7Y77R7oPeWSAO0NiLVINEyo/view?usp=sharing

Move the .zip file to the mmdetection working directory and unzip it:

```
mkdir inference_scripts
mv inference_scripts.zip ./inference_scripts
cd inference_scripts/
unzip inference_scripts.zip
cd ..
```

Move scripts to proper locations:
```
cp inference_scripts/mask-rcnn_r50_fpn_dental.py ./mmdet/.mim/configs/_base_/models
cp inference_scripts/coco_instance_dental.py ./mmdet/.mim/configs/_base_/datasets
cp inference_scripts/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_dental.py ./projects/ConvNeXt-V2/configs/
```

Run the following command to perform model inference:
```
python demo/image_demo.py inference_scripts/demo_images projects/ConvNeXt-V2/configs/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_dental.py --weights inference_scripts/epoch_20.pth --device cpu --out-dir convnext_v2_demo_outputs
```

**inference_scripts/demo_images** is the path to test images. Inference results will be saved at the folder **./convnext_v2_demo_outputs**.
