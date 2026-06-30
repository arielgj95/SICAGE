# SICAGE User Study

This folder contains the user-study specific pipeline:

- `prepare.py`: selects clips, renders generated variants, and writes `study_manifest.json`
- `server.py`: serves the local browser study and stores participant ratings
- `analyze.py`: computes descriptive statistics, significance tests, and plots
- `comparison_video_prepare.py`: creates side-by-side comparison videos for qualitative inspection
- `common.py`: shared constants and path helpers

Run all commands from the `SICAGE` root.

## 1. Prepare Study Assets

```bash
python -m user_study.prepare \
  --dataset-root "$PLAYLISTS_ROOT" \
  --output-dir "$REPO_ROOT/user_study_outputs/study_v1" \
  --clip-duration-sec 30 \
  --intro-per-culture 2 \
  --trials-per-culture 8 \
  --device cuda:0 \
  --no-culture-run-dir "$NO_CULTURE_RUN" \
  --fishr-run-dir "$FISHR_MDM_RUN" \
  --adversarial-run-dir "$ADV_MDM_RUN"
```

Outputs:

- intro videos
- trial videos for `real`, `no_culture`, `fishr`, and `adversarial`
- `user_study_outputs/study_v1/metadata/study_manifest.json`

## 2. Launch the Local Study Website

```bash
python -m user_study.server \
  --manifest "$REPO_ROOT/user_study_outputs/study_v1/metadata/study_manifest.json" \
  --host 0.0.0.0 \
  --port 8000 \
  --mode balanced_single_condition
```

Open `http://localhost:8000/` for a local dry run.

Modes:

- `balanced_single_condition`: 32 videos total, balanced across conditions
- `all_conditions`: all condition variants for all sequences

Results are saved under:

- `user_study_outputs/study_v1/results/participants/*.json`
- `user_study_outputs/study_v1/results/all_ratings.csv`

### 2.1 Open a Public Portal with Cloudflare Tunnel

For a temporary external participant link, install `cloudflared` using Cloudflare's official instructions for your OS:

- https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Keep the study server running locally and, in a second terminal from `SICAGE/`, open a quick tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare will print a random HTTPS URL such as `https://example-name.trycloudflare.com`. Restart the study server with that URL and a token that you will share only with participants:

```bash
export STUDY_PUBLIC_URL="https://example-name.trycloudflare.com"
export STUDY_TOKEN="replace-with-a-long-random-string"

python -m user_study.server \
  --manifest "$REPO_ROOT/user_study_outputs/study_v1/metadata/study_manifest.json" \
  --host 0.0.0.0 \
  --port 8000 \
  --mode balanced_single_condition \
  --public-url "$STUDY_PUBLIC_URL" \
  --access-token "$STUDY_TOKEN" \
  --secure-cookies \
  --trust-forwarded-for
```

Then share:

```text
https://example-name.trycloudflare.com/?token=replace-with-a-long-random-string
```

If you need a stable hostname instead of a random `trycloudflare.com` URL, use a managed Cloudflare Tunnel and set `--public-url` to that fixed HTTPS address.

## 3. Analyze Collected Ratings

```bash
python -m user_study.analyze \
  --results-dir "$REPO_ROOT/user_study_outputs/study_v1/results"
```

Outputs are written to:

- `user_study_outputs/study_v1/results/analysis`

Including:

- `descriptive_by_condition.csv`
- `descriptive_by_culture.csv`
- `descriptive_by_condition_and_culture.csv`
- `significance_tests.json`
- PNG plots

## 4. Create Side-by-Side Comparison Videos

```bash
python -m user_study.comparison_video_prepare \
  --dataset-root "$PLAYLISTS_ROOT" \
  --output-dir "$REPO_ROOT/comparison_outputs/study_material" \
  --clip-duration-sec 20 \
  --min-sequence-duration-sec 20 \
  --sequences-per-culture 10 \
  --device cuda:0 \
  --no-culture-run-dir "$NO_CULTURE_RUN" \
  --fishr-run-dir "$FISHR_MDM_RUN" \
  --adversarial-run-dir "$ADV_MDM_RUN"
```
