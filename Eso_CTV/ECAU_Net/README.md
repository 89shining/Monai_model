# ECAU-Net (3D) Reproduction

This folder contains a practical 3D ECAU-Net style implementation with training/testing flow aligned to your existing models.

## Files
- `ecau_net_3d.py`: network architecture (U-Net + cross-stage attention + ECA channel attention)
- `train_ecau_net.py`: 5-fold training + best-fold test
- `test_ecau_net.py`: standalone test using best fold from `fold_summary.csv`

## Run
```bash
cd D:\project\Monai_model\Eso_CTV\ECAU_Net
python train_ecau_net.py
# or rerun only testing
python test_ecau_net.py
```
