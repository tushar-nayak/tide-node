# Alignment check: atlas space vs affine registration

399 cohort transitions (79 patients) scored; status counts {'ok': 399}.
Region: target brain (eroded 2 mm) minus both tumor masks dilated by 5 mm.
Patient diff = mean over patients of (affine - atlas), 95% bootstrap CI over patients.

| metric | atlas (step 2) median | affine median | patient diff [95% CI] | transitions improved | atlas < 0.9 / affine < 0.9 |
| :--- | ---: | ---: | :---: | ---: | :---: |
| ncc_T1 | 0.566 | 0.848 | +0.245 [+0.220, +0.271] | 100% | 376 / 258 |
| ncc_FLAIR | 0.498 | 0.792 | +0.262 [+0.233, +0.288] | 97% | 386 / 323 |
| ncc_T2 | 0.466 | 0.830 | +0.344 [+0.316, +0.372] | 99% | 387 / 295 |
| brain_dice | 0.961 | 0.972 | +0.020 [+0.016, +0.023] | 70% | 17 / 0 |
| shift_mm | 2.537 | 0.575 | -2.805 [-3.151, -2.496] | 94% | - |

Link to step 2 (399 transitions, Spearman):
* atlas T1 NCC vs persistence WT Dice: +0.46
* registration gain in T1 NCC vs persistence WT Dice: -0.23

Worst atlas alignment (T1 NCC):

| patient | last -> target | atlas | affine |
| :--- | :--- | ---: | ---: |
| Patient-055 | week-017 -> week-039 | 0.003 | 0.386 |
| Patient-080 | week-001 -> week-018 | 0.123 | 0.177 |
| Patient-090 | week-000-2 -> week-029 | 0.167 | 0.404 |
| Patient-031 | week-184 -> week-190 | 0.169 | 0.437 |
| Patient-052 | week-002 -> week-089 | 0.187 | 0.531 |
| Patient-091 | week-001 -> week-014 | 0.190 | 0.338 |
| Patient-031 | week-002 -> week-015 | 0.192 | 0.300 |
| Patient-073 | week-001 -> week-015 | 0.198 | 0.639 |
| Patient-030 | week-001 -> week-017 | 0.213 | 0.251 |
| Patient-040 | week-015 -> week-028 | 0.214 | 0.432 |
