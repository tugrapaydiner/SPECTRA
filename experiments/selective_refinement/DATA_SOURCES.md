# Data attribution and exposure

All four official test partitions were used in earlier SPECTRA work. No new
independent test population is claimed by this continuation. Original feature
extraction is outside the runtime timer. New model fitting uses development-role
rows only; calibration rows are held apart by exact-feature groups.

The delivered arrays retain the original numeric features and labels. They are
repackaged as NPZ/raw uint8 files for reproducibility; new development/calibration
indices are provided separately, without rewriting official train/test partitions.
Original archive hashes and recovered-container file identities are retained.

Official UCI citations and licenses, verified September29,2026:

- Slate, D. (1991). Letter Recognition. DOI10.24432/C5ZP40.
  https://archive.ics.uci.edu/dataset/59/letter+recognition
- Alpaydin, E. and Alimoglu, F. (1996). Pen-Based Recognition of Handwritten Digits.
  DOI10.24432/C5MG6K. https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits
- Srinivasan, A. (1993). Statlog (Landsat Satellite). DOI10.24432/C55887.
  https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite
- Alpaydin, E. and Kaynak, C. (1998). Optical Recognition of Handwritten Digits.
  DOI10.24432/C50P49. https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits

Each is listed under Creative Commons Attribution4.0 International:
https://creativecommons.org/licenses/by/4.0/
SPECTRA's implementation license does not relicense the data. Pendigits and
OptDigits describe different training/testing contributors, which specifically
prevents assuming that random training-role calibration will automatically certify
performance on those test writers. Per-row writer IDs are not reconstructed from
feature arrays, and exact-feature grouping is not a substitute for such IDs.
