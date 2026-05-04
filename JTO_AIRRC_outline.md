# JTO outline — RONALD vs AirRC

Source synchronized from:
- `/Users/amrukwa/Downloads/ronald_CHEST.docx`

This outline now reflects two things at once:
1. what is already written in the current CHEST-format DOCX
2. how to convert that draft into the intended JTO / AirRC-centered paper

---

## Current source-draft message

The current DOCX argues:
- bronchovascular bundle removal improves nodule retention relative to simple thresholding
- RONALD handles low-dose CT anatomy better than crude intensity-based filtering
- preserving nodules attached to vessels/airways is clinically important

The current DOCX does **not** yet make the strongest JTO argument.

The stronger JTO paper should argue:
- **supervised segmentation is not automatically better when ground truth is scarce or mismatched**
- low-dose CT with lesion-adjacent anatomy is exactly the setting where this matters
- an anatomy-driven unsupervised method can be more aligned with the downstream diagnostic task

---

## Core thesis for the JTO version

**When dense task-matched ground truth is scarce, an unsupervised anatomy-guided segmentation method may outperform a supervised deep learning comparator by better preserving lesion-relevant bronchovascular anatomy and more plausible vessel structure.**

---

## What the DOCX already gives us

## Established clinical motivation
- early-stage lung nodules are hard to distinguish from adjacent vessels and airway walls
- low-dose CT adds noise, lower contrast, and acquisition variability
- CAD support is clinically motivated because screening volumes are large and radiologist time is limited

## Established method narrative
- RONALD is a full bronchovascular bundle pipeline
- preprocessing includes lungs, lobes, and mediastinum
- downstream stages separately model airways and vessels
- design goal is not just segmentation completeness, but preservation of diagnostically important nodules

## Established quantitative baseline
Current retention numbers from the DOCX:
- **Pomeranian:** 1270 / 1271 retained after RONALD vs 654 / 1271 after thresholding
- **DLCS:** 174 / 174 retained after RONALD vs 92 / 174 after thresholding

These should remain in the JTO package, but as supporting baseline context rather than the central comparison.

---

## Proposed manuscript logic for the JTO version

## 1. Introduction

### 1.1 Why bronchovascular segmentation matters
- early-stage nodules frequently contact vessels and sometimes airway walls
- this reduces visibility in screening LDCT
- segmentation quality should therefore be judged partly by whether lesion tissue is preserved rather than absorbed into anatomical masks

### 1.2 Why low-dose CT is hard
Content already present in the DOCX can be reused here:
- noise
- lower contrast
- variability across screening programs
- peripheral airway difficulty
- small-vessel difficulty
- lesion-vessel similarity

### 1.3 Why supervised learning may fail here
This is the major reframing step.
- supervised models inherit the assumptions and omissions of their labels
- lesion-adjacent anatomy is exactly where labels are most difficult and most clinically important
- benchmark overlap is not the same as downstream task utility

### 1.4 Study objective
Recommended close:
> We evaluated whether an unsupervised, anatomy-guided bronchovascular bundle segmentation pipeline could better preserve lesion-relevant anatomy and biologically plausible vessel organization than a supervised AirRC-derived comparator in low-dose CT.

---

## 2. Methods

## 2.1 Study design
- retrospective technical comparison
- RONALD as the unsupervised method
- AirRC-derived predictions as the supervised comparator
- thresholding retained as historical baseline only

## 2.2 Data
Use the existing DOCX dataset text as base:
- Pilot Pomeranian Lung Cancer Screening Program
- DLCS

For the direct Ronald-vs-AirRC analysis, add the exact comparison subset explicitly:
- likely Moltest1 / paired cases used by `compare_airrc_ronald_moltest1.py`
- state separately from the broader thresholding cohorts if needed

## 2.3 RONALD pipeline
Pull from the DOCX and keep concrete:
- lung segmentation
- lobe segmentation
- mediastinum localization
- airway-core extraction and modelling
- vessel extraction
- skeleton cleanup logic
- adaptive smoothing
- nodule-preserving behavior as design intent

## 2.4 AirRC comparator
State clearly:
- AirRC is a supervised annotation/inference resource
- it is valuable as a structured supervised baseline
- this study is not testing whether AirRC is a good resource in general
- this study tests whether its outputs are aligned with the specific lesion-preservation task

## 2.5 Evaluation framework

### 2.5.1 Historical baseline: thresholding retention
Keep the DOCX results here briefly:
- thresholding removes many lesion-containing structures
- RONALD preserves almost all nodules

### 2.5.2 Main comparison: nodule-in-BVB rate
Primary Ronald-vs-AirRC question:
- how often do annotated lesion coordinates fall inside the predicted BVB mask?

Interpretation:
- **higher inclusion is worse** for this task
- if the mask absorbs the lesion, diagnostically useful tissue is lost together with the bundle

### 2.5.3 Main comparison: lobar vessel plausibility
Primary question:
- how often do connected vessel components cross lobar boundaries in a biologically implausible way?

Interpretation:
- more cross-lobe components imply lower anatomical plausibility
- this is one of the strongest biology-aware endpoints

### 2.5.4 Qualitative analysis
Use figure panels showing:
- lesion-adjacent vessel region
- AirRC mask overlay
- RONALD mask overlay
- if possible, thresholding shown only as supporting baseline

---

## 3. Results

## 3.1 Thresholding baseline retained from the DOCX
This should stay concise.

Result statements already supported by the DOCX:
- **Pomeranian:** retention improved from `51.46%` to `99.92%`
- **DLCS:** retention improved from `52.87%` to `100%`
- Wilcoxon signed-rank test reported significance for both cohorts

## 3.2 Ronald-vs-AirRC lesion preservation
Fill from comparison outputs.

Result template:
> AirRC incorporated annotated lesion coordinates into the predicted bronchovascular mask more often than RONALD (`X/Y [%]` vs `A/B [%]`), indicating lower preservation of diagnostically relevant lesion tissue.

## 3.3 Ronald-vs-AirRC vessel plausibility
Fill from lobe-connectivity outputs.

Result template:
> AirRC showed more cross-lobe connected vessel components than RONALD, suggesting lower biological plausibility of the predicted vascular topology.

## 3.4 Qualitative examples
Pick 1-2 strongest cases:
- lesion adjacent to a vessel
- AirRC absorbs more lesion tissue
- RONALD preserves lesion boundary better

---

## 4. Discussion

## 4.1 Main interpretation
- RONALD is not claimed to be universally superior because it is unsupervised
- it may be superior **for this task** because the task is poorly captured by available supervised labels
- task alignment matters more than benchmark orthodoxy

## 4.2 Why this matters clinically
- preserving lesion visibility is more useful than maximizing anatomical mask extent
- in medical imaging, the best segmentation is often the one that best supports the clinical downstream objective

## 4.3 Why vessels deserve emphasis
The DOCX already supports this direction:
- nodules often feed on or contact vessels
- lesion-vessel adjacency is central to screening interpretation
- airway analysis remains relevant, but vessel behavior should lead the paper’s claim

## 4.4 Limitations
- no perfect voxel-level gold standard for the exact downstream task
- AirRC transfer may depend on cohort differences
- coordinate-based lesion analysis is clinically relevant but not identical to classical segmentation benchmarking
- thresholding baseline remains useful but should not dominate the JTO framing

## 4.5 Conclusion
Recommended endpoint:
> In low-dose CT, when dense task-matched ground truth is limited or imperfectly aligned with the diagnostic objective, an anatomy-guided unsupervised segmentation pipeline may outperform a supervised comparator by better preserving lesion-relevant anatomy and more plausible vascular organization.

---

## Figure plan for the JTO version

## Figure 1. RONALD pipeline overview
Keep from the DOCX.

## Figure 2. Representative preprocessing outputs
Compress current lungs/lobes/mediastinum figures if needed.

## Figure 3. Thresholding vs RONALD lesion-retention example
Use one strong supporting example only.

## Figure 4. AirRC vs RONALD lesion-adjacent comparison
This becomes the main qualitative figure.

## Figure 5. Lobe-colored vessel connectivity comparison
Use output from the lobe postprocessing workflow.

## Figure 6. Quantitative summary
- thresholding baseline bar/summary
- Ronald-vs-AirRC lesion inclusion
- Ronald-vs-AirRC cross-lobe vessel components

Because JTO allows **6 total figures/tables**, Table 3 from the DOCX may need to be merged into a figure or compact table panel.

---

## Concrete conversion checklist

- [x] Sync the markdown logic to the current `ronald_CHEST.docx`
- [x] Preserve thresholding-retention numbers from the DOCX
- [ ] Insert actual Ronald-vs-AirRC comparison numbers
- [ ] Insert actual lobe-connectivity counts
- [ ] Reduce figure/table count to JTO limits
- [ ] Rewrite abstract to strict JTO structure and length
- [ ] Normalize wording to BVB / LDCT / RONALD / AirRC

---

## One-sentence paper message

**RONALD already shows that anatomy-aware bronchovascular bundle removal preserves nodules better than simple thresholding; the JTO paper should extend that result by showing that, under scarce or imperfect ground truth, this unsupervised strategy can also outperform a supervised AirRC-derived comparator on lesion preservation and vessel plausibility.**
