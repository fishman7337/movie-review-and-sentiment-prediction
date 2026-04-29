# Data Card

## Dataset

The project expects the course-provided movie review dataset.

Recommended local path:

```text
data/raw/Movie reviews.csv
```

The original notebook used:

```text
./Dataset (B)/Movie reviews.csv
```

## Expected Columns

The reusable package expects at minimum:

- `Review`: text review content.
- `Score`: numeric score used for regression and sentiment labelling.

The original notebook also uses language and assignment-specific columns during cleaning.

## Preprocessing Summary

The original workflow includes:

- Missing score removal.
- Duplicate row removal.
- Very short review filtering.
- Score inversion so higher values mean more positive sentiment.
- DBSCAN-assisted sentiment binning.
- Language distribution checks.
- Lowercasing, punctuation removal, stopword removal, and lemmatisation.
- Shared tokenizer fitting and sequence padding.

## Risks and Limitations

- Data augmentation can introduce noisy paraphrases.
- Translation and paraphrasing models may change outputs between runs.
- Language detection can be brittle on short reviews.
- Regression scores are difficult to predict from short subjective text.

## Versioning Recommendation

Store dataset versions outside Git and record:

- File name.
- Source.
- Date received.
- Row count.
- Column schema.
- Checksum.
