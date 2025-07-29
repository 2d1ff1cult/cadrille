# `cadrille`: Multi-modal CAD Reconstruction with Online Reinforcement Learning
## All credits go to the original developers🤗

### Some notes from fork:
Ran into OOM errors with original train script, the one here has been modified

## Old notes
View old versions of 2d3dgen_cadrille.py via the notebook [here](https://colab.research.google.com/drive/1SbPwzw1lmNEslnP4IQ-oNY9zkoXzJ1uj#scrollTo=tBdBSUPmR_f2)

# Usage
The `chat_cadrille.py` script is the biggest difference from the original repo. Simply run the py script (assuming you've cloned [JPL-Su2025-2d3dgen](https://github.com/2d1ff1cult/JPL-Su2025-2d3dgen/))

To infer Cadquery scripts for a single mesh (stl or ply):
`py -3.10 chat_cadrille.py --mode 3d --mesh <path to mesh>`

To infer Cadquery scripts for a folder of meshes:
`py -3.10 chat_cadrille.py --mode 3d --mesh-dir <path to mesh folder>`

Previously, it was used as a chatbot, but due to fine-tuning, it has lost that capability. **Still working out how to best carry on with the inference**

## Train
To start training run *train.py* script:
```shell
python train.py --mode pc_img --use-text
```

During the development process at JPL, a run lasting 50000 steps was done. The model's weights can be downloaded here: [checkpoint-50000](https://drive.google.com/file/d/1BruYqOSxopopnFzmamtf7sndgatXcf2p/view?usp=sharing). For some metrics, a cross-entropy loss of 0.20 was achieved.

At evaluation (see below for more information), the following metrics were gathered:
## `checkpoint-50000`
mean iou: 0.822 median cd: 0.227

## `maksimko123/cadrille`
mean iou: 0.823 median cd: 0.203

To disable some of the modalities set *--mode* to *img* or *pc*, or disable *--use-text*. We don't provide RL fine-tuning code for now. Alternatively both [SFT](https://huggingface.co/maksimko123/cadrille) and [RL](https://huggingface.co/maksimko123/cadrille-rl) models can be downloaded from :hugs: HuggingFace.

## Inference
To predict CadQuery codes run *test.py* script:
```shell
python test.py --split deepcad_test_mesh --mode pc
```
### This creates a folder called `tmp_py` that stores a bunch of Cadquery scripts inferred from a desired dataset

## **If re-testing, make sure that the `tmp_py` folder in ./cadrille is empty**
To run on other datasets and modalities use *--split fusion360_test_mesh* or set *--mode* to *img* or *text*.

## Evaluation
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

```shell
@article{kolodiazhnyi2025cadrille,
  title={cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning},
  author={Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, Danila Rukhovich},
  journal={arXiv preprint arXiv:2505.22914},
  year={2025}
}
```
