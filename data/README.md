# Synthetic data generation

All experiments operate on procedurally generated synthetic datasets.  Each
episode constructs a mixture of datasets with independent seeds so that the
teacher sees a wide variety of domain shifts:

* **Classification:** Gaussian blobs with controllable margin, label noise, and
  dataset size imbalance.
* **NER:** Synthetic token sequences sampled from a simple grammar with entity
  spans injected at configurable frequencies.
* **Detection:** Small binary images containing geometric shapes with bounding
  boxes.

Datasets are generated on-the-fly; no external assets are required.  Running the
CPU debug configuration is sufficient to materialise probe sets and train the
students.
