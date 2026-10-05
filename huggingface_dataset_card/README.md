---
pretty_name: TED4C-L
license: other
license_name: ted4c-l-composite-rights
license_link: LICENSE
language:
- hi
- it
- ja
- tr
- en
size_categories:
- 100K<n<1M
tags:
- sicage
- ted4c-l
- co-speech-gesture-generation
- gesture-generation
- multimodal
- motion
- audio
- text
- lmdb
gated: false
---

# TED4C-L

TED4C-L is the multimodal dataset introduced with **SICAGE: Speaker-Independent
Culture-Aware Gesture Generation using TED4C-L Dataset** (ECCV 2026). It was
created to study co-speech gesture generation across four source groups: Indian,
Italian, Japanese, and Turkish TED speakers.

Paper: [SICAGE: Speaker-Independent Culture-Aware Gesture Generation using TED4C-L
Dataset](https://arxiv.org/abs/2606.30001), published in the ECCV 2026 proceedings.

Each record is a five-second window containing aligned motion, speech, and text
representations. Windows overlap with a stride of 0.5 seconds. The public release
contains derived numeric features, but it does **not** contain raw video, raw audio,
readable subtitles, or transcript strings.

## At a Glance

| Property | Value |
| --- | ---: |
| Usable video | 106.45 hours |
| Samples | 659,454 overlapping windows |
| Speakers/source videos | 764 |
| Window length | 5 seconds |
| Window stride | 0.5 seconds |
| Motion rate | 15 FPS before VQ-VAE encoding |
| Cultures/source groups | India, Italy, Japan, Turkey |
| Download size | approximately 189 GiB |
| Format | LMDB with pickled Python dictionaries |

The culture label is a **dataset-level source grouping**, based on the collection
and language organization used to build TED4C-L. It is not a verified label of a
person's nationality, ethnicity, identity, or cultural beliefs. It must not be used
to infer those attributes for individuals.

### Distribution by source group

| Label | Samples | Speakers/source videos |
| --- | ---: | ---: |
| Indian | 225,575 | 191 |
| Italian | 145,217 | 194 |
| Japanese | 116,056 | 196 |
| Turkish | 172,606 | 183 |
| **Total** | **659,454** | **764** |

### Speaker-independent split

The provided split is randomized **at the speaker/source-video level**, not at the
window level. Within each source group, the builder randomly shuffled speaker/video
identifiers and assigned approximately 80% to training, 10% to validation, and 10%
to testing. It then placed every window from a given identifier in the same subset.
The saved release is therefore one fixed randomized split with no speaker/video
overlap between subsets.

| Split | Samples | Speakers/source videos |
| --- | ---: | ---: |
| Train | 531,089 | 609 |
| Validation | 58,523 | 75 |
| Test | 69,842 | 80 |

Because adjacent windows overlap heavily, do not reshuffle individual windows across
subsets. Doing so can place nearly identical neighboring windows in both training and
evaluation data. Use the provided speaker-level split, or create another split by
assigning complete speakers/videos rather than individual samples.

## What Is in One Sample?

Every LMDB value is a pickled Python dictionary. Arrays are stored in the following
layout:

| Field | Stored shape | Meaning |
| --- | --- | --- |
| `motion` | `(512, 25)` | VQ-VAE motion representation; the SICAGE loader returns `(25, 512)` |
| `audio_mels` | `(64, 156)` | log-mel spectrogram; the loader returns `(156, 64)` |
| `audio_onsets` | `(156,)` | audio onset strength |
| `audio_wav2vec` | `(50, 1024)` | multilingual Wav2Vec2 speech representation |
| `text_features` | `(768,)` | LaBSE embedding of the aligned source-language subtitle text |
| `text_features_translated` | `(768,)` | LaBSE embedding from an available English subtitle track; otherwise a copy of `text_features` |
| `translation_source` | string | one of `translation_disabled`, `english_subtitles`, or `empty_text` in this release |
| `culture` | string | source-group label |
| `speaker` | string | source video identifier used as the speaker/domain identifier |
| `language`, `text_language` | string | source and subtitle language metadata |
| `sample_start`, `sample_end` | integer | window boundaries in source-video frame coordinates |
| `motion_fps`, `scene_fps` | float | processed motion and source-scene frame rates |

TED4C-L uses LMDB rather than the tabular Hugging Face `datasets` format, so load
it with LMDB or the SICAGE data loader instead of `datasets.load_dataset()`.

### Text Features in This Release

The public LMDB retains numeric LaBSE features and provenance only. The readable
source and English subtitle strings used to calculate those features were removed
before publication. `text_features` contains the source-language subtitle embedding.
`text_features_translated` uses an available English subtitle track when present;
otherwise it copies `text_features` or stores an empty-text embedding for windows
without aligned subtitles.

The SICAGE dataset loader uses `text_features` by default. Translated features are
selected only when `use_translated_text=True` or the corresponding command-line
option is set.

## Files

```text
data.mdb
lock.mdb
metadata/
  culture_encodings.pkl
  language_encodings.pkl
  metadata.pkl
  speaker_encodings.pkl
  subject_to_domain_indices.pkl
  training_languages_id_map.pkl
  training_speakers_id_map.pkl
  whole_dataset_splits_subject_independent.pkl
```

`data.mdb` contains 659,454 records. The files under `metadata/` contain keys,
integer encodings, and the official speaker-independent split.

## Download

Install `huggingface_hub` with Xet support, then download the repository snapshot:

```bash
python -m pip install -U "huggingface_hub[hf_xet]>=0.36.2,<1.0"
export TED4CL_DATASET_ROOT="/abs/path/to/TED4C-L"
```

```python
import os
from huggingface_hub import snapshot_download

dataset_root = os.environ["TED4CL_DATASET_ROOT"]
dataset_root = snapshot_download(
    repo_id="ariel-95/TED4C-L",
    repo_type="dataset",
    local_dir=dataset_root,
)
print(dataset_root)
```

Allow roughly 189 GiB for the download and additional working space for experiments.
The main file is very large, so interrupted downloads may take time to recover.

## Read the LMDB Directly

Install `lmdb`, then inspect a record as follows:

```python
import os
import lmdb
import pickle

dataset_root = os.environ["TED4CL_DATASET_ROOT"]
env = lmdb.open(
    dataset_root,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

with env.begin(write=False) as transaction:
    cursor = transaction.cursor()
    if cursor.first():
        sample_key = cursor.key().decode("ascii")
        sample = pickle.loads(cursor.value())
        print(sample_key)
        print({name: getattr(value, "shape", type(value).__name__)
               for name, value in sample.items()})
```

Python pickle must only be loaded from a trusted source. The SICAGE codebase provides
the training loader and split utilities in `dataset.py`. For the repository scripts:

```bash
export FULL_DATASET="/path/returned/by/snapshot_download"
export FULL_META="$FULL_DATASET/metadata"
```

## Intended Use

TED4C-L is intended for non-commercial research on:

- co-speech gesture generation and evaluation;
- multimodal speech, text, and motion representation learning;
- reproduction and comparison of the SICAGE experiments; and
- analysis of model behavior across the four dataset source groups.

Do not use TED4C-L for biometric identification, speaker surveillance, nationality or
ethnicity inference, decisions about individuals, or claims that the four labels
represent cultures in their full diversity.

## Limitations and Responsible Use

- TED speakers are public figures in staged talks and are not representative samples
  of national populations or everyday communication.
- Source-group labels are broad proxies and may encode language, recording, topic,
  demographic, and production differences in addition to gesture patterns.
- Overlapping windows are highly correlated; sample count is not equivalent to the
  number of independent observations.
- Motion, speech, and text features may retain information about identity, language,
  topic, or other personal attributes even though raw media is not included.
- Text embeddings can reflect transcription or alignment errors in the locally
  obtained source subtitle tracks. This release contains no readable subtitle or
  transcript strings.
- Models trained on this dataset can reproduce stereotypes or source imbalances.
  Report results by group and document any additional filtering.

For correction or removal requests, open a discussion in the
[SICAGE repository](https://github.com/arielgj95/SICAGE).

## License and Source Rights

The Hugging Face metadata uses `license: other` because TED4C-L combines
SICAGE-authored dataset organization, metadata, splits, and documentation with
numeric representations derived from TED source material. The complete rights
notice is in [`LICENSE`](LICENSE).

To the extent the SICAGE authors own the dataset selection, organization,
dataset-specific metadata, split definitions, and documentation, those contributions
are made available under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Users are responsible for making sure their use of any source-derived representation
is permitted by the applicable rightsholders.

## Citation

If you use TED4C-L or SICAGE, please cite:

```bibtex
@inproceedings{gjaci2026sicage,
  title={SICAGE: Speaker-Independent Culture-Aware Gesture Generation using TED4C-L Dataset},
  author={Gjaci, Ariel and Sgorbissa, Antonio and Murino, Vittorio},
  booktitle={European Conference on Computer Vision},
  pages={392--411},
  year={2026},
  organization={Springer}
}
```
