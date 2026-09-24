# Data preparation

## Source acquisition and input availability

The [official MSC project](https://parl.ai/projects/msc/) distributes the dataset through ParlAI. The reproducible preparation route below uses the public [nayohan/multi_session_chat mirror](https://huggingface.co/datasets/nayohan/multi_session_chat), pinned to revision `78b67491c43823fc169cab827ab3f82805e0235b`. This mirror is a separate distribution of MSC. Follow the dataset's applicable terms independently of HasMem's code license.

Install the optional data dependency and run:

```bash
pip install -e ".[data]"
python -m hasmem.preparation.msc --download-source /path/to/msc-source --output-dir data/msc
```

The command verifies SHA-256 hashes of the downloaded train and validation Parquet files, preserves their row order in local JSONL exports, and records the URLs, revision, row counts, and file hashes in `source_metadata.json`. Existing source files must match the pinned export. The default preparation then checks every selected source record and the complete ordered model inputs against the paper cohort manifests. The train and validation inputs match all published hashes: 4,526 / 272 owners and 12,698 / 662 records before token filtering. No dataset text is distributed in this repository.

The original JSONL export's byte hashes and canonical trace identities are recorded in [msc_provenance.json](msc_provenance.json). JSONL serialization can change file bytes; the verified model-input hashes cover the exact ordered fields consumed by HasMem. Native ParlAI files are not accepted directly. For an existing compatible export, use `--source-dir /path/to/msc-source` in place of `--download-source`.

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

The pinned mirror preparation and preserved-trace import produce the same verified model inputs. The Qwen2.5-7B token filter yields 268 owners, 646 records, and 535 questions from 272 owners and 662 records. Exact published model outputs additionally require the original trained checkpoint and tokenizer revision.

For an existing copy of the preserved paper traces, place `GLOBAL_PROMOTION_TRACE_TRAIN.jsonl` and `GLOBAL_PROMOTION_TRACE_VALIDATION.jsonl` in one directory and run:

```bash
python -m hasmem.preparation.msc --trace-dir /path/to/paper-traces --verify-paper-traces --output-dir data/msc
```

This checks their saved SHA-256 hashes and retains only the ordered fields consumed by HasMem. The output directory must be new. Trace files and raw dataset text are not distributed in this repository.

## LongMemEval-S

Use the released 500-question `longmemeval_s_cleaned.json`. Histories are ordered by parseable timestamps (source order is retained if dates cannot be parsed), rendered with dates and roles, and split into character-lossless chunks of at most 128 tokenizer tokens. Question dates are included in the query. The full history is retained before memory maintenance and retrieval.

The loader selects by question-ID hash and constructs memory before accessing answers. Questions sharing any haystack session are grouped into connected components for grouped statistics. The 500-question paper evaluation uses `lme_questions=500`; smaller counts are diagnostics and should be reported as subsets.

Data source: [MSC](https://parl.ai/projects/msc/) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval). Follow each source's license and access terms.
