# Final capacity-model lock — before official-test prediction

All eight source models have now been refit exactly once on their complete official
training partitions with random_seed 20260929 and thread_count=8. No official test
prediction from these final models has been made before this record.

MODEL_LOCK SHA256:
24cfcf4d4cbab39da249a20cf49a9ae461d3adae7aba34d7723b96d8508ce5b4

Selected source-model identities:
- Letter selected 512xdepth8 lr=.06
  CBM 93f5f02c7f80ad6ad83c14d5b779cc2873ae83fd200669f31be072685eb05f52
  JSON b638c2aa829396125bd69b12748feaea413110cd9f19b1124a9740aba1913b5b
- Pendigits selected 512xdepth8 lr=.06
  CBM b9eacb226ee4fc1cc13be8b595bfca47048635e80894a0e579ff5061ec051eb0
  JSON 01dad5c1c51fbe4ec8fe7c56553788a4228b9d6b7167ca64db1e3b7029839c0e
- Satellite selected 512xdepth7 lr=.06
  CBM 82b82ce5968de2e42da275400abcb5eb9249503737538c9c5ca89944758a4a6d
  JSON ffe90369d79c6bee25befcf0632bb1213f4293017b9940b32ebae6bd577873d6
- OptDigits selected 512xdepth7 lr=.06
  CBM 0d0bffce7cb6e1b5ba96a8919cbb4e2f8154cd499c86acf481d3deae7a080d3c
  JSON e4c3a3c64966c9391689e523f0de3bc9718c1e34ac05320d7e5ecd6c1737818c

Matched 256xdepth6 controls are also frozen in the local MODEL_LOCK. The complete
model files, exported C++ and fit receipts remain local evidence until packaging.

Next operation is one descriptive official-test evaluation of all eight locked
models. No capacity/learning-rate/depth change follows those predictions.
