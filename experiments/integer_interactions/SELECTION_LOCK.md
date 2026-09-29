# Complete selection lock after development pilot

The complete pilot is retained: full NCA lowers the best Letter validation count
1410->1386/1500; Pendigits1494->1495/1499; Satellite816->809/887. Within-class
whitening improves Letter1410->1418 but harms Satellite. Thus the first pilot does
NOT establish transferable learned-interaction value. No official test was read.
The first process was stopped by the tool timeout after21/24 fits; a complete
unchanged24-fit rerun is separately retained, without replacing favorable cells.

Proceed once with equal hyperparameter selection across uniform, diagonal-NCA,
full-NCA and supervised-whitening, not choosing a winning family retrospectively.
The full-NCA arm remains PRIMARY; whitening is an established simpler control.
Keep the original requirement of >=0.5-point improvement on at least two tasks
and <=1.25x cost. A control winning does not turn primary failure into success.

Three grouped splits:611,977,1543, cap fitting6000/validation2000, same as parent.
Each arm has C1/10/100 and gamma0.5/2/8/32 on Letter/Pendigits, or2/8/32/128 on
Satellite. The equally extended range is fixed now because some pilot optima
were at the upper boundary. Pick aggregate validation correct fraction, then
fewer summed support vectors, then C/gamma enumeration order. Each arm gets12
choices per split,432 total fitting cells. Full refits use all original training
rows and metric seed20260928. No model-selection/optimizer change after this lock.
Primary and three controls all evaluated, even where their validation is worse.
Any optimizer iteration-limit event is recorded, not called convergence.

This is continued development on already consumed public benchmarks, not a
fresh application or new held-out confirmation. Interpretation must say so.
