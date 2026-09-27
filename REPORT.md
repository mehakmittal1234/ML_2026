# Business Entity Resolution (Amazon ML Challenge 2026): model report

**Leaderboard (public):** v8_cal 0.982143, then **v14e 0.984861** (+0.0027). All other numbers in this report are either validation
scores (train fold 0, with labels) or label-free estimates. Estimates are marked as such.

---

## 1. Summary

| | Score | Kind |
|---|---|---|
| v8_cal (starting point) | 0.982143 | leaderboard |
| v14e_frfull (current best) | **0.984861** | leaderboard |
| v15_same (v14e + 6,055 pairs) | about 0.9850 | estimate |
| Validation, same pipeline (fold 0) | 0.98941 | labelled |
| Validation with every decision perfect on the pairs we retrieve | 0.99577 | labelled ceiling |

The model is a LightGBM cascade: blocking and retrieval, a stage-1 pair model, and a stage-2 context model, followed by test-set
corrections. Most of the leaderboard gain came from one discovery. **The score chain rejects certain true copies on the test set
only.** Names decorated with "Service", "& Associés", "Et Fils", "Cie" and similar get p near 0 on test, although the same pairs are
99.9% true in train. We found these pairs with a score-free structural model. We confirmed them without test labels in three ways:
the generator's per-source copy cap, count invariance, and decoration rates per entity. Adding them back gave v14 (+0.0027 over
v8_cal on the leaderboard).

Reaching 0.99 is not realistic with this data. About 0.004 of the remaining loss comes from address-less copies of businesses that
share a name, which cannot be told apart. About 0.001 to 0.0015 comes from copies that cannot be retrieved at a reasonable candidate
budget. The rest is spread thinly: each further fix is worth about 0.0001 to 0.0003 (section 10).

---

## 2. Task and metric

- For every Source-1 (S1) business, list the Source-2 and Source-3 records (S2, S3) of the same business.
- The score is macro F0.5, averaged per S1 entity. Precision is weighted 4 times more than recall. An S1 entity with no true match
  scores 1 when its list is empty.
- One false match on an entity with 4 true copies costs about 0.17 of that entity's score. One missed copy costs about 0.06.
  **Adding a pair helps only if it is more than about 74% likely to be true.**
- Test S1 entities: 1,732,544 (US 663,106, India 809,986, France 259,452). France does not appear in train.

## 3. What we learned about the data generator

These facts come from train labels (and from test counts where marked). Several later methods depend on them.

| Fact | Evidence | Used in |
|---|---|---|
| Each hidden entity is copied into S2/S3 with independent noise. Within a source, copies share an address "version". Names have no source layer. | train | source-version features (s56) |
| True copies per S1 entity: 3.46, the same for US and India; 5.6% of S1 entities have no match | train; leaderboard probe "all empty" 0.056 | count invariance |
| **Copy cap: at most 5 copies per entity in S2 and 6 in S3** | train maximum | cap test (section 7) |
| Name noise adds or swaps a word from a small fixed list. US/India: "center" 0.104, "services" 0.077, "service" 0.036, "partners" 0.029 per S1 entity | train | decoration rates (section 7) |
| **Planted same-street distractors have a HIGHER house number than their S1**: US 99.0%, India 85%. For true copies it is about 50/50 | train fold 0 | sign features (s67), same-name restores (v15) |
| IDs and row order are random, in train and in test | train labels; test checked with confident matches | no leak |
| Test has many more distractor records: about 40% of S2/S3 records, against 26% in train | count invariance per country | all test corrections |
| Content-identical records within a source always belong to the same entity (43,441 of 43,441 groups) | train | checked: v13 is consistent in 37,762 of 37,770 test groups |

## 4. Pipeline (baseline v8_cal, kept as the backbone)

| Step | Scripts | What it does |
|---|---|---|
| 1 | s00–s02b | Convert TSV to parquet; learn Indic-script dictionaries from train pairs; normalise names and addresses (legal forms, state abbreviations, digit-for-letter typos) |
| 2 | s03b, s03c | Blocking keys (name, address and number families) |
| 3 | s04d–s06 | Record-side ranker that keeps the top 60 candidate S1 per record |
| 4 | s07 | Retrieval: keep the top 8 S1 per S2/S3 record |
| 5 | s08, s09 | Pair features; stage-1 LightGBM, cross-fitted so train scores are out-of-fold |
| 6 | s11, s16, s17 | Stage-2 features: record competition, S1 context, sibling agreement, raw-name and house-number relations, name support, edit-type tells |
| 7 | s14b, s19 | Stage-2 LightGBM, plus a specialist for borderline pairs |
| 8 | s18 | Per-country prior-shift recalibration |
| 9 | s13 / efd.py | Decision: each record goes to its best S1 if the score is at least 0.7. Later outputs use an expected-F0.5 decoder instead |

Validation fold 0 is `hash(id1, seed 20260925) % 8 == 0`, with the official macro F0.5.

## 5. Validation: where the loss sits

**Scores on fold 0** (276k S1 entities, 955,924 true pairs):

| Setting | Macro F0.5 |
|---|---|
| Our pipeline (threshold 0.7) | 0.98941 |
| With the expected-F0.5 decoder | 0.98952 |
| Perfect, except our own decisions on ambiguous records | 0.9963 |
| Perfect on every retrieved pair (retrieval recall 0.9860) | 0.99577 |
| Every true pair except class A | 0.99365 |

**Where the 0.0106 of validation loss goes:**
- **Ambiguous records: about 0.0037.** These are records with no address whose name is shared by several S1 entities (104,549
  records; 13,139 true pairs in fold 0). We assign 2,624 of them (2,375 correct, 249 wrong) and abstain on the rest.
- **Everything else: about 0.0069.**
  - 9,082 true pairs are never retrieved.
  - 4,020 true pairs score below the threshold.
  - 2,531 true pairs lose their record to another S1.
  - 1,235 false pairs are accepted (889 distractors, 346 records owned by another S1).

**Retrieval misses** (13,336 true pairs in fold 0):
- 9,521 are address-less copies, mostly of businesses sharing a name. Nobody can resolve these.
- 3,815 have an address (India 3,212, US 603). Their names are scrambled ("Lyraonyxevo"), run together ("tglfoodprivate") or
  shared by up to 300 S1 entities, and their addresses are often cut down to "house number, city".

## 6. Why test is harder than validation

1. **About twice as many distractors per entity.** Train has 26% distractor records; test has 37–41%. Many are planted
   neighbour businesses: the same name on the same street with a slightly higher house number.
2. **France is never seen in training.** France has crowded addresses and same-name S1 neighbours (0.36 per entity).
3. **The score chain fails on test for some true-copy patterns** (section 7). This was the largest problem we found.

Model-implied F0.5 per country (s24) was US 0.98588, India 0.98599 and France 0.97981, against about 0.9899 on validation.

## 7. What we did, in order

### 7.1 Expected-F0.5 decoder (earlier session)
Chooses, per entity, the number of top candidates that maximises expected F0.5 (a Poisson-binomial calculation over candidates).
Validation 0.98941 → 0.98952 (+0.0001).

### 7.2 Test-time adaptation: v11–v13 (not scored on the leaderboard)
- **Neighbour-business drop rules** (s27–s43): found test-only clusters of same-name neighbours in the US and France.
- **Hybrid correction** (s53/s54): count invariance per cell (country × name relation × house-number relation × anchor) sets how many
  pairs to keep; a density-ratio model (train vs test) decides which ones. It removed 27,519 accepted pairs.
- **Near-twin specialist** (s62): trained on real generator neighbours (copies of a same-name S1 twin). Cross-validated AUC 0.998
  in India and 0.988 in the US.
- **v13_ens_efd:** a blend of the hybrid and specialist scores, the decoder and a France rule. 5,765,507 matches. Estimated
  0.9837 (range 0.9818–0.9839).
- **In hindsight:** back-solving from v14e's leaderboard score puts v13 near 0.982–0.983, the low end of its range. The census in
  7.6 shows why: the hybrid removed about 8,400 planted neighbours correctly, but it also dropped about 1,800 true lower-number
  copies and a similar number of true higher-number ones.

### 7.3 Candidate set (organiser FAQ)
The organisers said they rank on candidate-set size as well as score: "smaller candidate sets per Source 1 entity will be ranked
higher". A pruned later stage is allowed, as long as every matched ID appears in it. `s66_cand.py` writes `candidate_pairs.tsv`
from the stage-2 set: **7,674,729 pairs, 4.43 per S1 entity**. v8_cal's file was the full retrieval set, about 76M pairs or 44 per
entity, so this is about 10× smaller. It covers every matched pair of every output and passes the official validator.

### 7.4 New signal: direction of the house-number offset (s67)
No existing feature carried the sign of the house-number difference; all used the absolute value.

| Same street, nearby number | Copy's number higher than S1's |
|---|---|
| US planted distractors | 99.0% |
| India planted distractors | 85% |
| True copies | 50–51% |

On validation, a correction model with sign features improved log-loss from 0.01221 to 0.01148 and macro F0.5 from 0.98926 to
0.98951 against the same model without them. The feature matters most on test, where planted neighbours are about twice as common.
It was used in the census and in v15.

### 7.5 The key finding: true copies rejected on test only (v14)
1. **Structural model (s68).** A LightGBM on pair structure only (name, address, source-version, sign and street features), with no
   score-chain inputs. It is weaker than our scores (AUC 0.9949 against 0.9986), so we used it only as a detector: pairs it calls
   true (q ≥ 0.9) while our score says no (p < 0.5). On validation such disagreements are rare, and our score is right 90% of the
   time. On test they appear in large pockets: US 26,720, India 15,757, France 46,444.
2. **What those pairs look like.** Examples: "Rev It Up Coffee" / "Rev It Up Coffee Service", "Nyssa Esposito Muniholdings LLC" /
   "Nyssa Esposito LLC Service", same address, p ≈ 0 on test. The same pattern ("service" swapped into a US name at the same
   address) covers 30,412 train pairs: 99.9% true, p 0.999. On test, 15,353 such pairs get p = 0.32 (stage-1 p1 0.56), although
   every pair feature and ranker feature has the same distribution. The exact cause could not be traced: the old stage-1 model
   files were deleted during an earlier disk cleanup.
3. **Label-free checks:**
   - **Copy-cap test.** If the record's S1 already holds 5 confident S2 copies, the record cannot be one of its true copies.
     Distractors hit this about 0.3% of the time; true copies about 0.003%. Flagged US S2 pairs with q ≥ 0.97: 1 of 10,890.
     India: 0 of 4,669. Below q = 0.9 the rate is distractor level, so those pairs were not used.
   - **Count invariance.** Accepted pairs per test entity were below validation (US 3.329 against 3.372; India 3.340 against
     3.366). The additions bring them to validation level.
   - **Decoration rates.** US "service" copies per entity: expected 0.0350 (train truth × validation recall); v13 had 0.0043.
     "services", "center" and "partners" were already at their expected levels. So the failure was specific, not a general
     under-count.
4. **Outputs:**

   | Output | Pairs added | Total matches | Estimated gain | Leaderboard |
   |---|---|---|---|---|
   | v14_add | +32,801 (US 22,692; India 10,109): q ≥ 0.97, p < 0.5, record unassigned | 5,798,308 | +0.0010 to +0.0014 over v13 | not scored |
   | v14c_deco | +11,788 decorated copies, filled up to truth × recall per word; India q ≥ 0.97 only | 5,810,096 | +0.0004 to +0.00055 | not scored |
   | v14d_fr | +17,361 France decorated copies, capped at 0.030 per word | 5,827,457 | +0.0005 to +0.0007 | not scored |
   | **v14e_frfull** | +24,931 France decorated copies, whole pool | **5,835,027** | +0.0008 to +0.0010 | **0.984861** |

   France has no labels. Its decoration words were identified on test: "groupe", "france", "développement", "associés", "services"
   and "cie" all settle at about 0.029 per entity once the pool is added. That uniformity is what a fixed-rate generator
   decoration looks like. The cap test on the French pool: 4 of 12,637. The v14d and v14e outputs differ only in about 7,600
   "Et Fils" pairs; which one is better is untested.

### 7.6 Census by pair type and same-name restores (v15)
- **Census (s73).** For each cell (country × name relation × legal-form relation × house-number relation × decoration), compare
  accepted pairs per entity on test against train truth × validation recall.
  - The strongest shortfalls are US same-name, nearby-number cells: about 4,350 pairs, z from −10 to −18.
  - India names with extra words show the same pattern, smaller (about −500 and −400).
  - Across all cells, shortfalls and surpluses are about ±30,000 pairs, but within a cell they cannot be told apart.
- **Direction check** for US same-name, nearby-number pairs:

  | Accepted set | Lower number | Higher number |
  |---|---|---|
  | v9scal (before correction) | 12,418 | 20,520 (about 8,400 planted neighbours) |
  | Expected true copies | about 12,248 | about 12,150 |
  | v14e | 10,582 | 10,267 |

- **v15_same (s75):** +6,055 same-name copies that the test pipeline drops. The groups are:
  - a nearby number lower than S1's
  - a truncated number lower than S1's
  - a far or missing number with a unique S1 name
  - India address-less copies with a unique S1 name

  Cap test on S2: 2 of 2,618. 5,841,082 matches in total. Estimated +0.0001 to +0.0003 over v14e.

## 8. Label-free validation toolkit
We had no test labels and only a few leaderboard submissions, so every test-only change was checked with signals that do not need
labels:
- **Copy-cap test:** S2 ≤ 5 and S3 ≤ 6 copies per entity. Measures what share of an added set could be distractors.
- **Count invariance:** true copies per entity per pair type are a generator constant. Measured against train truth × validation
  recall.
- **Sign test:** true copies are symmetric in house-number direction; planted neighbours sit above the S1 number.
- **Decoration rates:** the generator applies each decoration word at a fixed rate per entity.
- **Gain estimate:** expected change in F0.5 per touched entity, computed at several assumed precisions. Break-even is about 74%
  precision.

Calibration against the leaderboard: v14e came in about 0.001 below its point estimate (0.9859–0.9867). Either v13's corrections were
worth less than estimated, or part of the additions (most likely the French "Et Fils" extras) is less precise than assumed.
**Treat every estimate in this report with a margin of about ±0.001.**

## 9. What did not work

| Idea | Result |
|---|---|
| Stage-3 retrain on all features (s51) | 0.98920–0.98941, no gain on validation |
| Positive-unlabelled posterior from scratch (s52) | fails the no-shift sanity check (0.98457) |
| Sibling-deviation stacking (s38/s39) | +0.00002, rejected |
| Non-exclusive decoding (a record listed under several S1) | 0.98947, against 0.98951 exclusive |
| Twin swap by exact house number (s63) | the model was right 462 times, the swap 34 times |
| Second retrieval by street word and number (s64) | recovers 70 of 3,815 missed pairs |
| Retrieval by rare address tokens (s76) | recovers 31–77 of 3,212 India misses |
| Retrieval by number + city and name substrings (s77) | recovers 519 of 3,815 inside 2.26M new candidates (0.02% true); worth about +0.0001 even if scored perfectly |
| Resolving ambiguous records by copy counts | the twin with fewer confirmed copies owns the record 58–71% of the time; below the 74% break-even |
| French street comparison (s74) | French accepted pairs have fully different streets 0.32% of the time, the same as US; 99% of such pairs are true on validation |
| Adding French one-word swaps at the same address (about 8,300 pairs) | cap violations 0.4–0.9%, distractor level; kept out |
| US address-less exact-name "excess" | a composition effect (more unique names on test), not distractors |
| Dictionary / ID / row-order leaks | none, on train and on test |

## 10. Why not 0.99

Test loss is about 0.015 (1 − 0.9849):
- **About 0.004** is address-less copies whose business name is shared by several S1 entities. There is no signal to split them
  beyond legal-form hints, which the model already uses. Every team pays this.
- **About 0.001 to 0.0015** is copies that are never retrieved: scrambled names with cut-down addresses, and crowded names. Recall
  ceiling on validation: 0.986.
- **The rest** is spread across model errors on test. The census puts cell-level mismatches at about ±30,000 pairs, but fixing them
  requires identifying individual pairs inside a cell. Each identifiable pocket is worth about +0.0001 to +0.0003.

A meaningful further jump would need a much stronger base model, for example a fine-tuned cross-encoder (≤ 8B parameters,
MIT/Apache licence) on a GPU. This environment has 4 CPUs and no GPU.

## 11. Files and submission guidance

The repository is `mehakmittal1234/ML_2026`, branch `claude/awesome-ride-rnl6x4`. Zipped `matching_results.tsv` files are in
`submissions/`:

| File | Matches | Status | MD5 of the TSV |
|---|---|---|---|
| `v14e_frfull_matching_results.zip` | 5,835,027 | **leaderboard 0.984861** | 1f7052312d78b1cd5942ca5fc3a5d35a |
| `v14d_fr_matching_results.zip` | 5,827,457 | not scored; ±0.0003 vs v14e | df8ad9c57bdf6c577599aa242bd325a1 |
| `v15_same_matching_results.zip` | 5,841,082 | not scored; estimated +0.0002 vs v14e | 091798c08669e9067a404c48a34fdc5c |

Only the private score of your best public submission counts, so a lower-scoring extra submission cannot hurt. If a submission is
left: v15_same.

Every output folder in `outputs/` also contains `candidate_pairs.tsv` (4.43 per entity) and passes the official validator.

## 12. Reproducing v14e / v15 from the stage-2 artifacts
1. Baseline to stage-2 scores: `run_upto_stage2.sh`, then the v9s / v9scal scoring (earlier session).
2. Test adaptation (v13): `s50_feats`, `s56_srcver`, `s53_dr`, `s54_hybrid` (hyb4), `s62_ns test` (ns1), a blend of the two
   (ens1) with `efd.py`, then `s55_build`, which gives `outputs/v13_ens_efd`.
3. Structural detector: `s67_sign feats train|test`, `s68_struct test st1`.
4. Additions:
   - `s70_add v13_ens_efd st1 ens1 0.97 US,India` gives v14_add
   - `s71_deco v14_add st1 ens1 US:0.8,India:0.97 0.9` gives v14c_deco
   - `s72_frdeco v14c_deco st1 ens1 full 0 1` gives v14e_frfull
   - `s72_frdeco … budget 0.030 0.9` gives v14d_fr
5. v15: the group analysis from s69 / s73 writes `work/analysis/v15_groups.parquet`; then `s75_samename v14e_frfull` gives v15_same.
6. Candidate file: `s66_cand v9scal <output dirs>`. Check with `utils/validate_submission.py --matching … --candidate …`.

Every script states its purpose, usage and measured RESULT in its docstring. Commit messages record the numbers.

## 13. Compliance notes (from the organiser FAQ)
- **Models:** LightGBM only. No pretrained model, no hosted LLM, and no model above 8B parameters in the pipeline. AI help with
  coding is allowed.
- **Data:** no external data. Normalisation uses small hand-written lists (legal forms, US/Indian state abbreviations, street
  types), which the FAQ explicitly allows. Indic dictionaries are learned from the provided train pairs.
- **Test-set use:** unsupervised statistics and structural checks on the test files; the FAQ explicitly allows these.
- **To fix before the final package:** several scripts loop over the literal country names (`('US', 'India')`, France excluded).
  The FAQ asks to treat country as an open set. Unseen countries already pass through unchanged, but the loops should become
  data-driven (countries present in both train and test).
- **Package layout:** the zip's top level holds only `output/`, `code/` and this methodology document. `requirements.txt` and
  `README.md` go in `code/business_entity_resolution/`.
