# Data card - Adult / Census Income

## Identity, source and license

- Dataset: Adult, OpenML name `adult`, version 2 (14 inputs and a binary target).
- Original source: Barry Becker's extraction from the 1994 US Census database.
- Creators: Barry Becker and Ronny Kohavi.
- Citation: Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
- Source and dictionary: https://archive.ics.uci.edu/dataset/2/adult
- License stated by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Access method: `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)`.
- The included raw snapshot retains OpenML's `class` column name. The processing view renames it `income`.

## Task, size and population

48,842 rows and 15 columns: six numerical and eight categorical inputs, plus
income (`<=50K`, `>50K`). The positive class is `>50K`. The records describe
historical US adults selected by the source extraction criteria: age >16,
AGI >100, fnlwgt >1 and hours worked >0. AGI is a source selection criterion,
not an additional predictor in this dataset. This is a selected population,
not a representative sample of all people or current workers.

Inputs are age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week
and native-country. `fnlwgt` is a census population weight; this lab retains it
as an input to match the required 14-feature setup. Training and metrics are
unweighted; they are not population estimates. Numerical and textual education
fields represent related information.

## Processing and quality

`?`/NA denotes a missing input. The raw snapshot is never cleaned in place.
After the stratified 70/30 split, exact complete-record duplicates are removed
from training only. Numerical medians, categorical modes, 1st/99th percentile
clipping, standardization and one-hot vocabularies are fitted on that training
set. Test data receives the same fitted transforms and retains every row.
The processed CSV contains 16 inputs (including two derived features), original
row IDs, a split identifier and the income label. Final split sizes, missingness,
duplicate counts, conflict counts and numerical model results are recorded in
`REPORT.md` and the results directory after execution.

`capital_net` is the difference between cleaned capital gain and capital loss;
`hours_per_age` is cleaned hours-per-week divided by cleaned age minus 16.
Clipping can suppress meaningful rare income-related values, especially capital
gains. Imputation assumes train modes/medians are a useful replacement and may
distort subgroup distributions. Range and category validity does not establish
the truth of an individual's income label. Feature-identical rows with conflicting
labels are reported without assuming which label is correct.

## Known biases and limitations

Historical sampling, population selection, class imbalance and unequal subgroup
representation can affect performance. Race, sex and nationality are sensitive
attributes; other features can act as proxies. The recorded categories are
limited historical encodings and should not be read as a complete description
of identity. Income reflects social conditions and cannot measure merit or
individual worth. A fixed nominal USD 50,000 threshold from the historical data
does not transfer directly to current economies or other countries.

Accuracy and F1 are aggregate metrics; this lab does not establish subgroup
fairness, causality, calibration or suitability for deployment. A random row
split can put identical feature vectors in both partitions. The number is
disclosed in the report; the prescribed experiment is not an evaluation with
independent feature groups. One split does not quantify statistical uncertainty.

## Privacy and intended use

No direct names or addresses appear among these 14 inputs. Nevertheless,
demographic and financial combinations may create linkage or re-identification
risk. Public availability does not eliminate privacy concerns. Do not attempt
to identify individuals, link records to named people, or use this educational
model for consequential decisions such as credit, employment or insurance.

Intended use: learning validation, preprocessing, leakage controls and
reproducibility. The source does not provide independent ground truth for
manual income verification. The 50-row human dictionary review is explicitly
pending in the supplied worksheet until completed by a person.

## Reproduction and maintenance

`config.yaml` fixes the seed, split, clipping and model settings.
`requirements.txt` fixes direct dependency versions; `requirements-lock.txt`
records the full tested environment when supplied. Use Python 3.11 for the
tested setup. `results/config_and_hashes.yaml` records actual versions, raw and
processed hashes and canonical hashes of encoded data. `verify.py` checks
saved model reloads, hashes and isolation of train statistics from test values.
Reruns preserve an existing human-review worksheet. Changes to source data,
versions or settings require fresh validation and updated evidence.
