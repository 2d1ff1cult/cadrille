# `cadrille`: Multi-modal CAD Reconstruction with Online Reinforcement Learning
## All credits go to the original developers🤗

### Some notes from fork:
Ran into OOM errors with original train script, the one here has been modified

# Usage
The `chat_cadrille.py` script is the biggest difference from the original repo. Simply run the py script (assuming you've cloned [JPL-Su2025-2d3dgen](https://github.com/2d1ff1cult/JPL-Su2025-2d3dgen/))

Previously, it was used as a chatbot, but due to fine-tuning, it has lost that capability. **Still working out how to best carry on with the inference**

## Train
To start training run *train.py* script:
```shell
python train.py --mode pc_img --use-text
```
To disable some of the modalities set *--mode* to *img* or *pc*, or disable *--use-text*. We don't provide RL fine-tuning code for now. Alternatively both [SFT](https://huggingface.co/maksimko123/cadrille) and [RL](https://huggingface.co/maksimko123/cadrille-rl) models can be downloaded from :hugs: HuggingFace.

## Inference
To predict CadQuery codes run *test.py* script:
```shell
python test.py --split deepcad_test_mesh --mode pc
```
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
