
---

# Deep transfer learning approach for automated cell death classification reveals novel ferroptosis-inducing agents in subsets of B-ALL. 

# First create the folders:

```sh
python3 create_folder_str.py
```

```sh
pip install - requirements.txt
```

### Mac / Windows
If using mac with mps replace in public_modul
```python
self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
```
with
```python
self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
```

## Folder structure
```
pycedei/ 
├── data/
├── logs/
├── results/
│   └── colored_img/
└── saved_models/
```

## Training

### Data must be placed under data/train/,  data/test, data/val ...
### Requirements was updated. 
```sh
python3 train_public.py
```
Model weights will be saved under saved_models and logs under logs.


## Inference

### For inference "train_model_resnet50_pawel_resnet50_ex1_ex42023_01_06__07_33_07.pth" must be placed in the saved_models folder.

```sh
python3 infer_public.py
```
Output CSV will be saved under results.

## Inference & Coloring
```sh
python3 color_public.py
```
Output Images will be saved under results/colored_img.

## Citation:
Pls cite and refer to when using the code: 
Paweł Stachura, Zhe Lu, Raphael M. Kronberg et al. Deep transfer learning approach for automated cell death classification reveals novel ferroptosis-inducing agents in subsets of B-ALL. , Cell Death and Disease, 2025

## Based on:
```latex
@article{werner2021deep,
  title={Deep transfer learning approach for automatic recognition of drug toxicity and inhibition of SARS-CoV-2},
  author={Werner, Julia and Kronberg, Raphael M and Stachura, Pawel and Ostermann, Philipp N and M{\"u}ller, Lisa and Schaal, Heiner and Bhatia, Sanil and Kather, Jakob N and Borkhardt, Arndt and Pandyra, Aleksandra A and others},
  journal={Viruses},
  volume={13},
  number={4},
  pages={610},
  year={2021},
  publisher={MDPI}
}
```

Note: The Python version and library dependencies have been updated to reflect more recent versions. These updates were made to ensure compatibility with current environments and may result in minor differences compared to the original implementation.
---
