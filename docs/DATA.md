# Data preparation

## Source acquisition and input availability

The [official MSC project](https://parl.ai/projects/msc/) distributes the data through ParlAI. Its documented inspection command is `parlai display_data -t msc`; `--include-last-session True` includes the final session. Follow the project's installation and dataset terms separately from this package.

This release consumes the split-specific JSONL export used by the original preprocessing scripts. Native ParlAI data is not directly accepted. The original export and the converter that produced its exact session/line ordering are not included, so native-release-to-export reconstruction has not been independently verified. A differently ordered export can change source record IDs. The published raw export hashes and verified trace identities are recorded in [msc_provenance.json](msc_provenance.json).

## Input schema

The MSC preparation command accepts one session per JSONL line. For example:

```json
{"dialog_id":"example-1","session_id":1,"persona1":["I collect stamps."],"persona2":["I live in Bristol."],"dialogue":["Hello there.","Nice to meet you."],"speaker":["Speaker 1","Speaker 2"]}
```

This example is constructed. The paper used the existing MSC split assignment. Keep all sessions for each dialogue in the same source split and preserve source line order. The owner identifier includes the split because dialogue IDs can be reused across native splits. The loader rejects overlapping prepared owner IDs across train and development data.

## Transformation

1. Normalize whitespace in source strings and cap profile values at 220 characters and dialogue values at 260 characters, matching the source preprocessing implementation.
2. Construct labeled profile facts from the first three entries for each speaker and dialogue facts from the first, middle, and last turns. Stable hashes determine fact ordering.
3. For the paper cohort, select the ordered source record IDs in `hasmem/preparation/cohorts/` and verify each reconstructed record hash. These manifests contain IDs and hashes, without source text or answers. They preserve the original preprocessing selection, which paired up to two profile/dialogue records per session, with distinct values and no normalized substring containment. Add owner and field labels to the record text; the answer remains the associated processed source value.
4. Group records by owner, order sessions chronologically, and deduplicate repeated record IDs within a session.
5. At model loading, retain records of at most 384 tokenizer tokens with answers of at most 48 tokens. Retain at most 48 events for training and 16 for development, then require at least two eligible events per owner.
6. Ask for the fields of the first and last selected records, using the most recent selected value when a field repeats. Duplicate target fields yield one question. Sort owners by a stable hash, independently of scores.

The default raw preparation fails on missing record IDs, mismatched record hashes, or a mismatched ordered model-input hash. `--new-cohort` explicitly creates a new cohort using the pairing rule; its results are a separate evaluation. The generated manifest records source SHA-256 hashes and preprocessing settings. `DATA_AUDIT.json` records tokenizer-dependent counts and final cohort hashes.

The paper trace identities and cohort membership were checked against the saved experiment audit. Importing those traces reproduces exactly the original model inputs; the Qwen2.5-7B token filter was rechecked and yields 268 owners, 646 records, and 535 questions from 272 owners and 662 records. End-to-end raw export preparation has not been rerun because the original export is unavailable locally. Exact published outputs also require the original trained checkpoint and tokenizer revision.

For an existing copy of the preserved paper traces, place `GLOBAL_PROMOTION_TRACE_TRAIN.jsonl` and `GLOBAL_PROMOTION_TRACE_VALIDATION.jsonl` in one directory and run:

```bash
python -m hasmem.preparation.msc --trace-dir /path/to/paper-traces --verify-paper-traces --output-dir data/msc
```

This checks their saved SHA-256 hashes and retains only the ordered fields consumed by HasMem. The output directory must be new. Trace files and raw dataset text are not distributed in this repository.

## LongMemEval-S

Use the released 500-question `longmemeval_s_cleaned.json`. Histories are ordered by parseable timestamps (source order is retained if dates cannot be parsed), rendered with dates and roles, and split into character-lossless chunks of at most 128 tokenizer tokens. Question dates are included in the query. The full history is retained before memory maintenance and retrieval.

The loader selects by question-ID hash and constructs memory before accessing answers. Questions sharing any haystack session are grouped into connected components for grouped statistics. The 500-question paper evaluation uses `lme_questions=500`; smaller counts are diagnostics and should be reported as subsets.

Data source: [MSC](https://parl.ai/projects/msc/) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval). Follow each source's license and access terms.
