# Sources and notices

This project derives its `src/morphmimic` training/environment/model core and ONNX export from Lizhi Yang et al., MorphMimic, commit 770394b4a0cad9effbf807a6c03435d1699733d0, Apache-2.0. Its license is preserved in MORPHMIMIC_LICENSE.

BlinkMimic (Fuyi Chen), InstantTrack (Huy-Quang-Dao), and supermimic_vibed (Chong Zhang, ETH Zurich) informed the optimizer, data-retarget and preview implementations. Local Bumi retarget helpers are used under their original MIT notices; preserve their source notices when redistributing them. No external source trees or assets are silently modified.

Robot geometry, controls, BONES-SEED and LAFAN1 remain in their original read-only directories. Code licensing does not grant data/asset rights. The prepared manifest records exact input and helper hashes. The original LAFAN1 dataset is licensed CC BY-NC-ND 4.0; its fixed-version terms are archived in docs/licenses/LAFAN1_license.txt. Keep dataset and code rights distinct when distributing artifacts.

The preview bundles a text subset of Noto Sans SC under SIL Open Font License1.1. Original notice and license: preview/static/OFL-NotoSansSC.txt. The font is served locally; no Google Fonts request is made by viewers. Media dependencies retain their installed aiortc/PyAV and bundled-library notices.

The LAFAN-to-SMPL rest-frame calibration uses the MIT-licensed General Motion Retargeting (GMR) LAFAN1 and SMPLX-to-G1 configuration conventions, https://github.com/YanjieZe/GMR. The original GMR notice is preserved in docs/licenses/GMR_LICENSE.txt. Missing LAFAN calibrations are explicitly masked.

MotionDecode references originate from Chingmu / CMRobot, https://huggingface.co/datasets/CMRobot/MotionDecode. Preserve the dataset attribution and its original research/prototype and non-commercial terms in any data redistribution. Its robot CSVs are not human ground truth.
