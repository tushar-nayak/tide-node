# Retrospective assessment review

Header checks: 371 current scans. Review sheets: 12.

No automated flag is a diagnosis or anatomical approval. Human review is pending; quality-admitted coverage remains 0%.
Only selected previous/current FLAIR/CT1 volumes and masks undergo voxel-value checks. Other modalities and unselected scans have header-only checks.

## Cases

- false_negative: Patient-023 / week-059 — [false_negative_Patient-023_week-059.png](false_negative_Patient-023_week-059.png)
- false_negative: Patient-089 / week-056 — [false_negative_Patient-089_week-056.png](false_negative_Patient-089_week-056.png)
- false_negative: Patient-072 / week-021 — [false_negative_Patient-072_week-021.png](false_negative_Patient-072_week-021.png)
- false_positive: Patient-019 / week-015 — [false_positive_Patient-019_week-015.png](false_positive_Patient-019_week-015.png)
- false_positive: Patient-054 / week-036 — [false_positive_Patient-054_week-036.png](false_positive_Patient-054_week-036.png)
- false_positive: Patient-029 / week-045 — [false_positive_Patient-029_week-045.png](false_positive_Patient-029_week-045.png)
- true_positive: Patient-073 / week-216 — [true_positive_Patient-073_week-216.png](true_positive_Patient-073_week-216.png)
- true_positive: Patient-032 / week-067 — [true_positive_Patient-032_week-067.png](true_positive_Patient-032_week-067.png)
- true_positive: Patient-061 / week-034 — [true_positive_Patient-061_week-034.png](true_positive_Patient-061_week-034.png)
- true_negative: Patient-003 / week-014 — [true_negative_Patient-003_week-014.png](true_negative_Patient-003_week-014.png)
- true_negative: Patient-025 / week-030 — [true_negative_Patient-025_week-030.png](true_negative_Patient-025_week-030.png)
- true_negative: Patient-029 / week-152 — [true_negative_Patient-029_week-152.png](true_negative_Patient-029_week-152.png)

Fill human_review.csv with alignment/segmentation/surgery findings, reviewer and date. Do not tune on these outcome-selected cases and call it independent validation.
