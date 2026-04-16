# DeepLabV3+ (MONAI-style pipeline)

This folder contains a DeepLabV3+ implementation and a training/testing flow aligned with `train_attention_unet.py`:

- `train_deeplabv3plus.py`: 5-fold training, select best fold, then test on `imagesTs/labelsTs`.
- `test_deeplabv3plus.py`: standalone testing from a specified checkpoint.
- `pipeline.py`: shared data/model/inference utilities.
- `networks/`: DeepLabV3+ network code (`deeplabv3_plus.py`, `xception.py`, `mobilenetv2.py`).

## Data layout

Uses the same layout as your MONAI scripts:

- `D:\project\Monai_model\data\imagesTr`
- `D:\project\Monai_model\data\labelsTr`
- `D:\project\Monai_model\data\imagesTs`
- `D:\project\Monai_model\data\labelsTs`

Image-label matching follows the `_0000` rule, e.g.:

- image: `case_001_0000.nii.gz`
- label: `case_001.nii.gz`

## Run

From this directory:

```bash
python train_deeplabv3plus.py
```

Standalone testing:

```bash
python test_deeplabv3plus.py --checkpoint ./runs/DeepLabV3Plus/fold_0/best_model_fold0.pth
```

