# Lar-Net: A Hierarchical Referring Image Segmentation Framework for Poisonous Weed Identification on the Tibetan Plateau
This repository contains the official PyTorch implementation for the paper: "**[Lar-Net: A Hierarchical Referring Image Segmentation Framework for
Poisonous Weed Identification on the Tibetan Plateau]**". 
![Lar-Net Logo](Lar-Net.png)
Our work introduces **Lar-Net**, a novel framework featuring an optimized hierarchical decoder for Referring Image Segmentation (RIS). Lar-Net addresses the limitations of standard Transformer-based decoders, such as spatial detail loss and insufficient vision-language fusion. By incorporating a **Deep Fusion** module, an **Enhanced Context Injector**, and a **Final Refinement** block, Lar-Net achieves state-of-the-art performance on both general benchmarks like RefCOCO and our specialized, challenging **Tibetan Plateau Poisonous Weed Segmentation Dataset (TPSD)**.

## Core Contributions

- **Hierarchical Decoder (Lar-Net)**: A coarse-to-fine decoder specifically designed for segmentation, preserving 2D spatial information and progressively refining masks.
- **Enhanced Fusion Mechanisms**: Novel components like the Deep Fusion Block and Enhanced Context Injector that enable robust integration of multi-scale visual details and global semantic context.
- **TPSD Dataset**: We constructed a new large-scale dataset for RIS, focused on the critical real-world challenge of identifying poisonous weeds in complex grassland environments. *(Note: This dataset is for internal validation and is not publicly released at this time.)*
- **State-of-the-Art Performance**: Lar-Net significantly outperforms baseline models on both RefCOCO and TPSD, demonstrating its effectiveness and generalization capabilities.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- PyTorch & Torchvision
- CUDA
- Other dependencies can be installed via `pip`:
  ```bash
  # It is highly recommended to create a virtual environment first
  # pip install virtualenv && virtualenv etris && source etris/bin/activate
  
  pip install -r requirements.txt
  ```
  *(Please see `requirements.txt` for a full list of dependencies like `numpy`, `opencv-python`, `loguru`, `pyarrow`, `lmdb`, `tqdm`, `segmentation-models-pytorch`, `albumentations`, etc.)*

### 2. Data Preparation

Our project utilizes two main types of datasets: the public benchmark RefCOCO and our custom TPSD. All datasets need to be converted into LMDB format for efficient training.

#### TPSD (Our Custom Dataset) The Tibetan Plateau Poisonous Weed Segmentation Dataset (TPSD) will be used for subsequent research and has not yet been publicly released.
#### RefCOCO

Please follow the standard procedures to download and preprocess the RefCOCO, RefCOCO+, and RefCOCOg datasets. This includes downloading MSCOCO images, the respective annotation files (`refs(unc).p`, etc.), and mask files. Convert all splits (`train`, `val`, `testA`, `testB`) into LMDB format and place them in `/home/featurize/work/data/lmdb/refcoco/`.

## Training

All training processes are initiated via `train.py` and configured using `.yaml` files in `config/`.

To train our final Lar-Net model on RefCOCO:
```bash
# Set CUDA devices, e.g., for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training script from the project root
bash run_scripts/train.sh
```
*(Note: `run_scripts/train.sh` should be configured to use the correct config file, e.g., `config/refcoco/bridge_r101.yaml`)*

To train on the TPSD dataset, create a new config file (e.g., `config/tpsd/lar_net.yaml`) and update the `train_lmdb`, `val_lmdb`, and `mask_root` paths to point to your prepared data.

## Evaluation

The `test.py` script is used for evaluating a trained model on a specific dataset split.

1.  **Configure the Test Split**: Open the relevant `.yaml` config file (e.g., `config/refcoco/bridge_r101.yaml`) and modify the `TEST` section:
    ```yaml
    TEST:
      test_split: testA  # or val, testB
      test_lmdb: /home/featurize/work/data/lmdb/refcoco/testA.lmdb # Ensure path matches split
      visualize: False
    ```

2.  **Run Evaluation**:
    - **To test the default `best_model.pth`** associated with the config:
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      python test.py --config config/refcoco/bridge_r101.yaml
      ```
    - **To test a specific weight file**:
      ```bash
      export CUDA_VISIBLE_DEVICES=0
      python test.py \
          --config config/refcoco/bridge_r101.yaml \
          --weight /path/to/your/specific_model.pth
      ```

## Citation

## Acknowledgements

This project is built upon the excellent codebase of [ETRIS](https://github.com/kkakkkka/ETRIS). We sincerely thank the authors for their valuable contribution to the community.
