# Assignment 2 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you. Write only what you
can actually explain.*

- **Name**: Nour Farhat
- **Student ID**: 19299 
- **Email**: nfarhat.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Business problem, and the cost of being wrong
# Assignment 2 Report

- **Name**: Nour Farhat
- **Student ID**:
- **Email**:
- **Group**: BBADBA 5A

> ✍️ = your words go here. Everything else is a fact from `train.ipynb`. Delete this line and every ✍️ note before submitting.

## Business problem, and the cost of being wrong

**Decision.** A regional general hospital with no neuroradiologist on site (e.g. in Tunisia) has its brain MRIs read by a general radiologist. The model sorts each scan into one of three outcomes:

| Outcome | What happens | Who acts |
|---|---|---|
| **pass** | normal report | local radiologist |
| **review** | sent to a remote neuroradiologist (teleradiology, paid per read) | remote neuroradiologist |
| **fail** | urgent referral to neurosurgery / oncology | referring doctor |

**Cost of being wrong:**
- **False negative** (a tumour passed as normal): ✍️ delayed diagnosis, worse prognosis, liability. Your estimate and why.
- **False positive** (a healthy scan sent for urgent referral): ✍️ an unnecessary consult and extra imaging. Your estimate and why.
- **Review:** ✍️ the fee for one remote read.

✍️ *Which mistake is worse, roughly by how many times, and why.*

**How this drives the pipeline:**
- **Score:** the decision is per scan, "is there any tumour?", so the model's score is `1 − P(notumor)`.
- **Metric:** recall on tumour scans and expected cost per 1,000 scans, not accuracy.
- **Thresholds:** chosen by sweeping the cost on the validation split, not left at 0.5.

## Dataset

- **Roboflow Universe:** workspace `kaplan-business-school-gqxef`, project `brain-tumour-detection-svrlh`, version `1`.
  <https://universe.roboflow.com/kaplan-business-school-gqxef/brain-tumour-detection-svrlh/dataset/1>
- **One image** = one 2D slice of a brain MRI, resized to 640×640, with no augmentation. Licence CC BY 4.0.
- **Classes (4):** `glioma`, `meningioma` and `pituitary` (three tumour types) and `notumor`.
- **Origin:** the uploader does not state where the images come from. They match the public Kaggle *Brain Tumor MRI Dataset*, which merges three public sources.

✍️ *Why this dataset over the others you looked at, and why it actually serves the hospital's decision. Reasons you can use: a real validation split for choosing thresholds, no pre-baked augmentation, and a `notumor` class that lets us count false positives.*

## What the dataset's composition implies

| Class | Train | Val | Test | Total | Share |
|---|---|---|---|---|---|
| glioma | 894 | 281 | 146 | 1,321 | 24.0% |
| meningioma | 928 | 281 | 124 | 1,333 | 24.2% |
| notumor | 1,003 | 248 | 155 | 1,406 | 25.5% |
| pituitary | 1,033 | 289 | 123 | 1,445 | 26.2% |
| **Total** | **3,858** | **1,099** | **548** | **5,505** | |

- **Balance:** every class is 24–26% of the data, and every class is split about 70/20/10.
- **Leakage check:** an exact-duplicate check (MD5 hash of each image) found 0 test images in train, 0 validation images in train, and 0 duplicates inside train.
- **Object size:** this is classification, so there are no boxes. A tumour usually covers a small part of the slice, and images are downscaled from 640 to 224 pixels for training.
- **Tumour rate:** 74.5% of scans in the dataset show a tumour, against an assumed 5% at the hospital. All costs are therefore computed at the hospital's base rate.
- **Thin numbers:**
  - Each test class has only 123–155 images, so one image moves a class's recall by about 0.7 points.
  - Every false-positive figure rests on 155 healthy test scans.
  - Differences of 1–2 points between classes are not meaningful.

✍️ *Interpretation: does the test split resemble the scans this hospital would see (one curated slice vs a full scan, different scanners and populations)? And what can you therefore NOT conclude?*

## Finetuning and baselines

- **Model:** `yolo11n-cls`, a classification model pretrained on ImageNet.
- **Training run:** Google Colab, T4 GPU, 20 epochs, `imgsz=224`, `seed=0`, `patience=10`, all other Ultralytics defaults. Training took 12 minutes.
- **Validation top-1 accuracy:** 0.986 at the final epoch.

✍️ *Why classification rather than detection (the decision is one per scan), and why the nano model.*

**Results on the same 548 test images:**

| Model | 4-class accuracy | Tumour vs no-tumour AUC | Best cost / 1,000 scans, model alone |
|---|---|---|---|
| **Finetuned yolo11n-cls** | **0.976** | **1.000** (0.99987) | **€3,817** |
| COCO yolo11n, zero-shot | n/a (no tumour classes) | 0.538 | €570,000 |
| Trivial: always "pituitary" (the training majority class) | 0.224 | 0.500 | €1,500,000 |

- **COCO zero-shot:** its score is the highest box confidence of any COCO object in the image. Its AUC of 0.538 is close to chance.
- **The cost column** uses the best threshold picked on test itself, so it is optimistic and only for comparing models. The threshold I ship is chosen on validation (below).

✍️ *What changed, and was the gap as large as you expected?*

## Per-class evaluation and failure modes

| Class | Test images | Precision | Recall |
|---|---|---|---|
| glioma | 146 | 0.986 | **0.959** |
| meningioma | 124 | **0.931** | 0.976 |
| notumor | 155 | 0.993 | 0.981 |
| pituitary | 123 | 0.992 | 0.992 |

**Confusion matrix (test; rows = true class, columns = predicted class):**

| | glioma | meningioma | notumor | pituitary |
|---|---|---|---|---|
| **glioma** | 140 | 6 | 0 | 0 |
| **meningioma** | 2 | 121 | 1 | 0 |
| **notumor** | 0 | 2 | 152 | 1 |
| **pituitary** | 0 | 1 | 0 | 122 |

- **Weakest recall:** glioma, at 0.959. All 6 of its errors were predicted as meningioma.
- **Weakest precision:** meningioma, at 0.931. It absorbs 6 gliomas, 2 healthy scans and 1 pituitary.
- **Tumour → no tumour:** 1 of 393 tumour scans (a meningioma) was predicted as healthy. This is the most expensive error.
- **No tumour → tumour:** 3 of 155 healthy scans were predicted as a tumour.

✍️ *Which class carries the headline number, which drags it down, and why glioma ↔ meningioma confusion matters less for THIS decision than tumour ↔ no tumour.*

## Cost matrix and operating point

| Item | Euros | Source / reasoning |
|---|---|---|
| Missed tumour (FN) | 30,000 | ✍️ |
| Unnecessary urgent referral (FP) | 600 | ✍️ |
| One remote neuroradiologist read | 50 | ✍️ |
| Base rate of tumours | 5% | ✍️ |

- **Formula:** `cost / 1,000 = 1,000 × [base × (P(pass | tumour) × FN + P(review | tumour) × review) + (1 − base) × (P(review | healthy) × review + P(fail | healthy) × FP)]`
- **Assumption:** the remote reader gets reviewed scans right.
- **Thresholds:** swept from 0 to 1 in steps of 0.01 on the validation split. Test was used once, to report.
- **Model alone (one threshold):** chosen threshold 0.63, which costs €7,494 per 1,000 test scans.
- **With a review band:** pass below **0.10**, review from 0.10 to 0.77, fail from **0.77**. Cost: **€5,349 per 1,000 scans**.
- **The curve:** `outputs/cost_curve.png`, and live in the dashboard.

**Sensitivity:** the band re-chosen on validation for other base rates and FN costs.

| Base rate | FN cost | t_low | t_high | Share reviewed |
|---|---|---|---|---|
| 1% | 30,000 | 0.63 | 1.00 | 1.6% |
| 2% | 30,000 | 0.10 | 1.00 | 5.2% |
| 5% | 5,000 | 0.63 | 0.77 | 0.6% |
| 5% | 30,000 | 0.10 | 0.77 | 3.1% |
| 10–20% | 30,000 | 0.10 | 0.77 | 2.6–2.9% |

- **t_high:** stays at 0.77 from a 5% base rate upwards. At 1–2% it moves to 1.00, meaning no scan goes straight to urgent referral.
- **t_low:** falls to 0.10 as soon as a miss costs about €15k or more at a 5% base rate.

✍️ *Which assumption would most change your answer if it were wrong, and why (hint: look at what moves in the sensitivity table).*

## What a person still has to check

- **Band:** pass < 0.10 ≤ review < 0.77 ≤ fail.
- **Share reviewed:** 3.1% of scans at a 5% base rate.
- **Workload:** at 20 brain MRIs a day, that is 0.6 reviews a day. At 15 minutes a read and 6 reading hours a day, it needs 0.03 neuroradiologists. That fits a pay-per-read teleradiology contract, not a hire.

| Option | Cost / 1,000 scans (test) | Share sent to a person |
|---|---|---|
| Do nothing (pass everything) | €1,500,000 | 0% |
| Person checks everything | €50,000 | 100% |
| Model decides alone (t = 0.63) | €7,494 | 0% |
| **Model + review band (0.10–0.77)** | **€5,349** | **3.1%** |

✍️ *Is 20 MRIs a day and 15 minutes a read realistic for this hospital? Is the review load realistic? Which option would you ship, and why?*

## Limitations & next steps

✍️ *Your real limitations, each with a concrete next step. Candidates from the results:*
- *One curated 2D slice per case, not a full MRI series → validate on full scans from the target hospital.*
- *The healthy images probably come mostly from one source → the model might learn the source, not the tumour; test on scans from a new hospital or scanner.*
- *The thin test set (123–155 per class, only 155 healthy) → a larger local test set, or bootstrap confidence intervals.*
- *Costs and base rate are assumptions → replace them with the hospital's real referral rate and teleradiology contract.*
- *The one missed tumour scored below 0.10 → look at that image to see why.*

## Generative AI use disclosure

✍️ *Adapt so it is exactly true for you, e.g.:* I used Claude (Anthropic) to compare candidate Roboflow datasets, write the code in `train.ipynb`, `costs.py` and `app.py`, debug the Colab setup, and help structure and format this report. The business framing, the cost figures, the choice of thresholds to ship, the interpretation of the results and the conclusions are my own, and I reviewed and can explain all of the code.
