# DDUnet (Paper-Style) for Esophageal CTV Auto-Segmentation

Files:
- `ddunet_model.py`: 2D DDUnet implementation (5-level U-Net + dilated context block with dilation 1/2/4)
- `train_ddunet.py`: 5-fold training on `imagesTr/labelsTr`
- `test_ddunet.py`: selects best fold by validation Dice and tests on `imagesTs/labelsTs`

## Run

```bash
cd D:\project\Monai_model\Eso_CTV\DDUnet
python train_ddunet.py
python test_ddunet.py
```

## Data naming

- image: `CTV_000_0000.nii.gz`
- label: `CTV_000.nii.gz`
- matching rule: remove `_0000` from image name to find label.

## Paper-aligned settings in this implementation

- 2D training/inference
- CT clipping to `[-150, 200]`
- resize each slice to `256x256`
- Adam (`lr=1e-4`, betas `0.9/0.999`, weight_decay `0`)
- 5-fold CV (80/20 per fold)
- Dice loss

Note: This is a faithful engineering reproduction of the method description, not the original authors' private source code.
