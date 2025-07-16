## `cadrille`: Multi-modal CAD Reconstruction with Online Reinforcement Learning

### Some notes from fork:
Ran into OOM errors with original train script, the one here has been modified

### Installation

Install Python packages according to our [Dockerfile](Dockerfile). We support DeepCAD (test), Fusion360 (test), Text2CAD (train / val / test), and CAD-Recode (train, val) datasets. Follow our [instruction](data/README.md) to download and preprocess data.

### Train

To start training run *train.py* script:
```shell
python train.py --mode pc_img --use-text
```
To disable some of the modalities set *--mode* to *img* or *pc*, or disable *--use-text*. We don't provide RL fine-tuning code for now. Alternatively both [SFT](https://huggingface.co/maksimko123/cadrille) and [RL](https://huggingface.co/maksimko123/cadrille-rl) models can be downloaded from :hugs: HuggningFace.

### Inference

To predict CadQuery codes run *test.py* script:
```shell
python test.py --split deepcad_test_mesh --mode pc
```
To run on other datasets and modalities use *--split fusion360_test_mesh* or set *--mode* to *img* or *text*.

### Evaluation

To evaluate IoU, invalidity ratio, and chamfer distance run *evaluate.py* script:
```shell
python evaluate.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/8b811b14-e646-48d6-9a0c-06a9655bdbaf" alt="cadrille scheme"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d6ae21f5-6c3c-4b7b-a2e9-ff0a310caa3d" alt="cadrille predictions"/>
</p>

