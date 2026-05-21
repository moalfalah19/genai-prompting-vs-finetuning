# Multilingual Fact-Checking Dataset

## Overview

This dataset consists of fact-checked claims collected from public fact-checking websites. Each instance contains a claim, its veracity label, the associated fact-check article text, and the language of publication. The dataset is intended for research on multilingual fact-checking and claim verification.

The data is split into training and validation

---

## Data Splits

### Training Set

**Label distribution**
- REFUTES: 1041  
- MIXED: 133  
- SUPPORTS: 44  

Total instances: 1218

**Language distribution**
- en: 340  
- es: 224  
- pt: 169  
- hi: 124  
- fr: 50  
- de: 47  
- fa: 40  
- ar: 24  
- pl: 24  
- ta: 24  
- mr: 22  
- te: 16  
- bn: 16  
- pa: 14  
- ur: 14  
- ml: 14  
- id: 13  
- bg: 11  
- kn: 10  
- fil: 8  
- ro: 6  
- da: 5  
- or: 2  
- hr: 1  

---

### Validation Set

**Label distribution**
- REFUTES: 130  
- MIXED: 13  
- SUPPORTS: 1  

Total instances: 144

**Language distribution**
- en: 42  
- es: 28  
- pt: 21  
- hi: 15  
- fr: 6  
- de: 5  
- fa: 5  
- ar: 3  
- ta: 3  
- pl: 3  
- bn: 2  
- mr: 2  
- te: 2  
- id: 1  
- bg: 1  
- pa: 1  
- ur: 1  
- ml: 1  
- kn: 1  
- fil: 1  

---

## Data Format

Each instance is represented as a JSON object with the following fields:

- **claim**: A short textual statement whose factual accuracy is evaluated.  
- **label**: The veracity label assigned by the fact-checking source. Possible values:
  - `REFUTES`: The claim is false.
  - `SUPPORTS`: The claim is true.
  - `MIXED`: The claim contains both accurate and inaccurate elements.
- **doc**: The full text of the corresponding fact-check article, including context, evidence discussion, and the final verdict.
- **lang**: ISO 639-1 language code indicating the language of the claim and article.

---

## Example Instance

```json
{
  "claim": "A fruit smoothie naturally can cure UTIs in 14 days",
  "label": "REFUTES",
  "doc": "... full fact-check article text ...",
  "lang": "en"
}
```

---

## Collection Process

Data was collected from publicly available fact-checking websites. For each claim, the associated article text was retrieved and aligned with its published verdict. Labels were normalized into three classes: REFUTES, SUPPORTS, and MIXED.

---
