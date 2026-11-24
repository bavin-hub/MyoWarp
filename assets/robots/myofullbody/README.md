# Whole-body biological model

Model assets are copied from the Apache-2.0 `musclemimic-models==1.0.6` package.
See LICENSE, NOTICE and the source headers for attribution. Original package:
https://github.com/amathislab/musclemimic_models

`myowarp/musclemimic/model.py` builds the checkpoint-compatible **1.0.5** variant
by applying MyoSuite 3.0.0's documented release-specific attribute values. It
removes finger joints/muscles, adds mimic sites, and sets muscle control ranges
to match the released 354-action policy. Load through this builder rather than
opening the unmodified source XML when evaluating `amathislab/mm-10m-2`.
