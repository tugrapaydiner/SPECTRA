# Model-selection freeze before final evaluation

Parent protocol: fc3b62627be2b30f34fda8310ea5012c73ac32b0.
No predictions on batches7-10 have been made at this freeze. Training uses1-5; validation uses6. No refit on6. These checkpoints are research artifacts, not gas-safety models.

Full local FREEZE.json SHA256: **ffe73d0278fe8c3bfa213320f2a47298b58f40a05267c720f5096b193fc25208**.
Original official archive SHA256: 91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4.
Scaler checkpoint SHA256: 3351e2a94bd1328294aeb96772611bc80abb7b5ce186b5a92adc2963678a28b0.
Training source identities: data.py=64d4092e284108c91eeb367367d7420cd24a840e00f434d29e02b7a0ff01e269; study.py=a0e34b1bf9e0b4d847ca580d24005053b5fdb66d2cc26fadf3b24b4d2ab01695.

| Family | Selected configuration | Validation balanced accuracy |
|---|---|---:|
| LinearSVC | C1 |0.7966398025026926|
| RBF SVC | C100, gamma0.1/128;237 supports |0.7843075608961403|
| MLP | hidden64, all seeds101/202/303 |0.7942506381796197|
| Nyström + LinearSVC |256 bases,gamma0.1/128,C10,all seeds101/202/303 |0.8144154490558534|

MLP/Nyström family values average three fixed seeds. Seed101 remains the predetermined deployment member, not the best observed seed. The full-SVM validation admission FAILS: below90% and does not exceed the simpler control. All selected families will still be evaluated on later batches as declared. No post-test reselection or relaxation of the gate.

All74 classifier fits and18 shared mapping preparations completed. Recorded fit/validation/checkpoint CPU cost totals295.817473633 seconds; complete training process wall303.397709608 seconds. These exclude subsequent export, compilation, test and benchmark costs. No GPU or pretrained weights.

Checkpoint identities:
- linear-c1:7a009a553e10e36c652d65ee67e36dd83e39dfcb50fb084597ed5d0dda224fb1
- svm-c100-g0.1:fdca5f783cfb995ac3d69cf536d0b5883f10debd52d84289a0d318d954567086
- mlp-64-s101:3e6ace148356799617591010c3ce710f64674cf433b949eb138a75ff1eb6c138
- mlp-64-s202:53e641615db5ad359b6b0ccb8f3b8f74be657f79907179cd06afcb875aa96f77
- mlp-64-s303:acb995bb245e3b7d2c4365e2bac59276e86c09c102842ef9b03ae58777bc6807
- nys-m256-g0.1-s101-c10:9db2f3cec5a4188f29964ac1f35e780aad6df8272038b99d70731f4acc1c47f9
- nys-m256-g0.1-s202-c10:8ffe3c778d007f3eae04d517ad71e805b14d141054aac2e276f70eb136df6d67
- nys-m256-g0.1-s303-c10:125417cfa9724fdd584d8d1fcccc19065a0a60d285ac9cfd1c0ac491c1ba8ff1

## Numerical implementation check fixed before final calls

The native-control prototype has36 passing synthetic/parser tests, including nonsymmetric Nyström normalization to catch transposition errors. Current native.py SHA256421b5d7e485387a1db7db50e5912880e35cc4f4e267645be47b4df2f763d362c.
Acceptance for nonbitwise linear/MLP/Nyström lowerings: every observed class must match the selected trained model, with score error <=1e-7+1e-6*abs(reference). This is a declared numerical comparison, not a universal arithmetic certificate. An implementation failure is retained and cannot be repaired by silently widening this threshold. The folded/unfolded baseline pair remains mandatory. No future measurements, success claim or release are implied.