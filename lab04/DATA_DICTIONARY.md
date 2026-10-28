# Adult data dictionary and 50-row review guide

Source: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult).
The original OpenML target `class` is called `income` in the processing view.
`?` is an allowed raw feature token denoting missingness, not an income label.
OpenML also represents missingness as NA. Numerical storage may be integral
floating point; non-missing numerical values must still be whole numbers.
Source row IDs in the audit are zero-based indices in the OpenML frame.

| Field | Expected type | Domain / check |
| --- | --- | --- |
| age | int | 17 to 90 inclusive |
| workclass | category | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked; ? / NA = missing |
| fnlwgt | int | Integer > 1 (source extraction criterion) |
| education | category | Preschool, 1st-4th, 5th-6th, 7th-8th, 9th, 10th, 11th, 12th, HS-grad, Some-college, Assoc-voc, Assoc-acdm, Bachelors, Masters, Prof-school, Doctorate; ? / NA = missing |
| education-num | int | 1 to 16 inclusive |
| marital-status | category | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse; ? / NA = missing |
| occupation | category | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces; ? / NA = missing |
| relationship | category | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried; ? / NA = missing |
| race | category | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black; ? / NA = missing |
| sex | category | Female, Male; ? / NA = missing |
| capital-gain | int | 0 to 99999 inclusive |
| capital-loss | int | 0 to 4356 inclusive |
| hours-per-week | int | 1 to 99 inclusive |
| native-country | category | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands; ? / NA = missing |
| income | category | <=50K, >50K |

## Education consistency

| education | education-num |
| --- | ---: |
| Preschool | 1 |
| 1st-4th | 2 |
| 5th-6th | 3 |
| 7th-8th | 4 |
| 9th | 5 |
| 10th | 6 |
| 11th | 7 |
| 12th | 8 |
| HS-grad | 9 |
| Some-college | 10 |
| Assoc-voc | 11 |
| Assoc-acdm | 12 |
| Bachelors | 13 |
| Masters | 14 |
| Prof-school | 15 |
| Doctorate | 16 |

## What to check in each sampled row

1. Check numerical values against the dictionary; verify no non-integral values.
2. Verify each present category belongs to the allowed domain.
3. Verify education matches education-num using the table above.
4. Treat missing workclass, occupation or native-country as a reported quality
   issue allowed by policy, not an impossible category or a reason to delete a row.
5. Flag contradictions to the stated dictionary; do not invent unsupported rules
   such as assuming a young adult cannot be married or receive capital income.
6. Confirm income has a valid category. The dictionary cannot establish that the
   label is factually correct; no access to an individual's original census record
   is provided. The separate conflict audit checks label consistency for identical
   feature vectors, not the truth of their labels.
7. Fill human_review_status with REVIEWED_OK or FLAGGED and record a short note.

The automated checks are an aid. Human review remains pending until a person
inspects and completes the 50-row worksheet. A reviewed sample is not proof of
quality for the full dataset.
