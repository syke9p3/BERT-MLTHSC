# Multilabel Classification of Tagalog Hate Speech using Bidirectional Encoder Representations from Transformers (BERT)

## Clone with git-lfs
Since this repo contains large data files (>= 50MB), you need to first download and install a git plugin called git-lfs for versioning large files, and set up Git LFS using command git lfs install in console, in order to fully clone this repo.

## Proponents: 
- Saya-ang, Kenth G. (@syke9p3)
- Gozum, Denise Julianne S. (@Xenoxianne)
- Hamor, Mary Grizelle D. (@mnemoria)
- Mabansag, Ria Karen B. (@riavx)

## Overview

This repository contains source files for the thesis titled, **Multilabel Classification of Tagalog Hate Speech using Bidirectional Encoder Representations from Transformers (BERT)**, at the Polytechnic University of the Philippines. The model classifies a hate speech according to one or more categories: Age, Gender, Physical, Race, Religion, and Others. 

Hate speech encompasses expressions and behaviors that promote hatred, discrimination, prejudice, or violence against individuals or groups based on specific attributes, with consequences ranging from physical harm to psychological distress, making it a critical issue in today's society. 

## Keywords
*Bidirectional Encoder Representations from Transformers; Hate Speech; Multilabel Classification; Social Media; Tagalog; Polytechnic University of the Philippines; Bachelor of Science in Computer Science*

## Labels

**Multilabel Classification** refers to the task of assigning one or more relevant labels to each text. Each text can be associated with multiple categories simultaneously, such as Age, Gender, Physical, Race, Religion, or Others.

| Label                                                        | Description                                                                                                      |
|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| ![Age](https://img.shields.io/badge/Age-FE5555)             | Target of hate speech pertains to one's age bracket or demographic |
| ![Gender](https://img.shields.io/badge/Gender-F09F2D)       | Target of hate speech is related to gender identity, sex, or sexual orientation |
| ![Physical](https://img.shields.io/badge/Physical-FFCC00)   | Target of hate speech pertains to physical attributes or disability |
| ![Race](https://img.shields.io/badge/Race-2BCE9A)   | Target of of hate speech pertains to racial background, ethnicity, or nationality |
| ![Religion](https://img.shields.io/badge/Religion-424BFC)   | Target of of hate speech pertains to affiliation, belief, and faith to any of the existing religious or non-religious groups |
| ![Others](https://img.shields.io/badge/Others-65696C)   | Target of of hate speech pertains other topic that is not relevant as Age, Gender, Physical, Race, or Religion |

# Dataset
2,116 scraped social media posts from Facebook, Reddit, and Twitter manually annotated for determining labels for each data split into three sets: 

| Dataset        | Number of Posts | Percentage |
|----------------|-----------------|------------|
| Training Set   | 1,267           | 60%        |
| Validation Set | 212             | 10%        |
| Testing Set    | 633             | 30%        |

# Results

The testing set containing 633 annotated hate speech data used to analyze performance of the model in its ability to classify the hate speech input according to different label in terms of Precision, Recall, F-Measure, and overall hamming loss.

| Label    | Precision | Recall | F-Measure |
|----------|-----------|--------|-----------|
| Age      | 97.12%    | 90.18% | 93.52%    |
| Gender   | 93.23%    | 94.66% | 93.94%    |
| Physical | 92.23%    | 71.43% | 80.51%    |
| Race     | 90.99%    | 88.60% | 89.78%    |
| Religion | 99.03%    | 94.44% | 96.68%    |
| Others   | 83.74%    | 85.12% | 84.43%    |

Overall Hamming Loss: 3.79%
