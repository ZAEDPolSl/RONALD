# JTO package for RONALD manuscript

Source synchronized from:
- `/Users/amrukwa/Downloads/ronald_CHEST.docx`

Last sync intent:
- align the BRONCO markdown notes with the current CHEST-format RONALD draft
- preserve the concrete dataset/method/result details from the DOCX
- keep JTO-specific adaptation notes explicit

## Current manuscript state from the DOCX

The current full draft is still framed primarily as:
- **RONALD vs simple thresholding / large-object filtering**
- core endpoint: **nodule retention after bronchovascular bundle removal**

The DOCX already contains:
- title-independent scientific narrative for the RONALD pipeline
- dataset descriptions for **Pilot Pomeranian** and **DLCS**
- literature review for airway and vessel segmentation
- quantitative retention results
- figure legends
- reference list

It does **not** yet center the manuscript on:
- AirRC as the main comparator
- supervised vs unsupervised learning under ground-truth scarcity
- lobe-aware vessel plausibility as a primary result

So the current writing basis is strong, but the framing is still pre-JTO-repositioning.

## Extracted structured abstract content from `ronald_CHEST.docx`

### Background
Lung cancer remains the deadliest cancer because diagnosis is often late. In screening CT, early-stage nodules can be obscured by vessels and airway walls, so explicit analysis of the bronchovascular bundle is important for efficient detection.

### Research Question
Will careful removal of the bronchovascular bundle from CT images make image analysis more efficient and increase diagnostic potential for lung cancer?

### Study Design and Methods
RONALD is a bronchovascular bundle segmentation pipeline for low-dose chest CT. It performs preprocessing with lung, lobe, and mediastinal localization, then separates vessel and bronchial-tree segmentation within the lung parenchyma. Evaluation used low-dose CT screening datasets including **DLCS** and the **Pilot Pomeranian Lung Cancer Screening Program**.

### Results
The current DOCX reports improved nodule retention relative to simple thresholding:
- **DLCS:** `52.87% -> 100%`
- **Pomeranian:** `51.46% -> 99.92%`

Table 3 values in the DOCX:
- **Pomeranian:** 1271 nodules, 654 retained after size filtering, 1270 retained after RONALD
- **DLCS:** 174 nodules, 92 retained after size filtering, 174 retained after RONALD

Statistical note from the DOCX:
- Wilcoxon signed-rank comparison vs thresholding reported as significant for both datasets
- p-value reported as `< 1e-6` for both datasets
- effect sizes reported as `0.867` (Pomeranian) and `0.870` (DLCS)

### Interpretation
The DOCX conclusion is that RONALD enables bronchovascular segmentation while preserving nodules relevant to early-stage lung-cancer diagnosis.

## Key scientific details preserved from the DOCX

## Clinical motivation
- early-stage nodules are often attached to vessels or airway walls
- this reduces conspicuity on low-dose CT
- segmentation of the bronchovascular bundle can simplify downstream diagnostic reading and CAD processing

## Datasets currently described in the DOCX

### Pilot Pomeranian Lung Cancer Screening Program
- years: 2009-2011
- total program size described: 2002 patients
- benchmark subset used: 1201 series
- intended role in the draft: main benchmark due to large number of annotated nodule locations

### Duke Lung Cancer Screening (DLCS)
- years described: 2015-2021
- total size described: 1613 patients
- includes 3D bounding-box nodule annotations and outcomes
- intended role in the draft: external/second cohort validation

## RONALD pipeline points explicitly present in the DOCX
- lung segmentation
- lobe segmentation
- mediastinum segmentation
- airway / tracheobronchial modeling
- vessel segmentation
- post-filtering aimed at preserving nodules rather than absorbing them into removed anatomy
- adaptive branch smoothing
- alternative skeletonization handling when Lee skeleton complexity becomes an outlier
- use of Kimimaro when bronchial-tree skeleton complexity is excessive

## Figures currently described in the DOCX
1. Processing pipeline for bronchovascular bundle modelling
2. Lung and lobes segmentation
3. Mediastinum segmentation
4. Finetuned tracheobronchial tree
5. Vessel segmentation with malignant nodule excluded from vesselness
6. CT before/after filtration with retained nodules highlighted
7. Number of bronchial-tree skeleton nodes with cutoff line

## JTO constraints to keep in mind
- structured abstract: **250 words max**
- headings: **Introduction or Hypothesis / Methods / Results / Conclusions**
- main text: **4000 words**
- figures/tables: **6 total**
- references: **50 max**
- keywords: **3-5**

## Recommended JTO adaptation path

### What to keep from the DOCX almost directly
- dataset descriptions
- literature review blocks on airway and vessel segmentation
- concrete retention numbers
- pipeline figures and pipeline wording
- biological argument that lesion-adjacent bundle structures matter

### What to change for the JTO version
- move from **thresholding comparison** to **unsupervised vs supervised under imperfect ground truth**
- use **AirRC** as the main external supervised comparator
- keep thresholding as historical baseline or supplemental context only
- foreground **vessel behavior** over broad airway discussion
- reframe success criterion around:
  - preserving lesion-relevant anatomy
  - avoiding absorbing nodules into the predicted bundle
  - maintaining biologically plausible vessel topology

## Current best JTO title options

### Preferred
**Can Unsupervised Segmentation Outperform Supervised Deep Learning When Ground Truth Is Scarce? A Case Study of Bronchovascular Bundle Segmentation on Low-Dose CT**

### Alternative
**Unsupervised Bronchovascular Bundle Segmentation on Low-Dose CT Under Ground-Truth Scarcity: A Case Study Against a Supervised AirRC Baseline**

## Immediate manuscript gap list
- AirRC comparison needs to replace thresholding as the primary external comparator
- result placeholders for Ronald-vs-AirRC need to be filled with actual numbers
- the current DOCX has 7 figures, so JTO compression to **6 total figures/tables** is required
- final abstract must be shortened and rewritten into JTO structure
- terminology should be normalized to **low-dose CT**, **bronchovascular bundle (BVB)**, and **RONALD** throughout

## Related local files
- `source/BRONCO/JTO_AIRRC_outline.md`
- `source/BRONCO/MOLTEST1_AIRRC_RONALD_COMPARISON.md`
- `source/BRONCO/compare_airrc_ronald_moltest1.py`
