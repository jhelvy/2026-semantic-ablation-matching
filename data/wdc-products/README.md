# WDC Products (80% corner cases, multi-class, 0% unseen)

Peeters & Bizer, "WDC Products: A Multi-Dimensional Entity Matching Benchmark" (EDBT 2024),
http://webdatacommons.org/largescaleproductcorpus/wdc-products/

The data files here are **not tracked in git**: the dataset page states no licence, so
nothing derived from record text is committed. Rebuild every file with:

    uv run scripts/01b_prepare_wdc.py

The script downloads `80multi.zip` (sha256 `5385e249...068981`), checks the hash, and writes
`data_original/{offers,catalog,wdc_mapping}.csv` and the three v1 transforms to `data_test/`.
Then `uv run scripts/02_prepare_factorial.py --dataset wdc-products --verify` writes the
factorial cells.

- Queries (`offers`): the 1,000 offers of `wdcproductsmulti80cc20rnd000un_gs.json.gz`.
- Catalog: one offer per product from `..._train_large.json.gz`, the one with the lowest `id` (500).
- Mapping `wdc_mapping.csv` (`id_offers`, `id_catalog`): each query to the catalog record with its
  `cluster_id`. Many-to-one (two queries per catalog record).
- Text: title only (`title_only=True`).
- The build is deterministic (sorted Ciphered Words map). It reads `/usr/share/dict/words`; the
  dictionary's sha256 is recorded in `data_test/wdc_build.json`.
