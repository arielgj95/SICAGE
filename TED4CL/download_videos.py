#!/usr/bin/env python3
import os
import re
import argparse
import html
import tempfile
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

import yt_dlp
import subprocess

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
CLIENT_SECRETS_FILE = str(BASE_DIR / "client_secret.json")
TOKEN_PATH = BASE_DIR / "token.json"

# -----------------------
# Defaults & constants
# -----------------------

# These playlists are private. If you want to use these you can ask for permission, otherwise use other personalized playlsits
DEFAULT_PLAYLIST_LINKS = [
    'https://www.youtube.com/playlist?list=PLZvzPFufocwvWtCs2hN1f8NmJXvnWaBoq',
    'https://www.youtube.com/playlist?list=PLZvzPFufocwtN0j7RZ60kNuIt-xnXEBC1',
    'https://www.youtube.com/playlist?list=PLZvzPFufocwsqpk2h_ntQeVUTjtXEttJb',
    'https://www.youtube.com/playlist?list=PLZvzPFufocwutvA0-TdHSXRm_M0vbLezd'
]

COOKIES_FILE = None
COOKIES_FROM_BROWSER = None
YTDLP_PROXY = None
YTDLP_IMPERSONATE = None
YTDLP_SLEEP_REQUESTS = 0.0
YTDLP_SLEEP_SUBTITLES = 0.0
ENGLISH_LANGUAGE_CODES = ("en", "en-US", "en-GB", "en-CA", "en-AU", "en-IN")
TRANSLATION_LANGUAGE_ALIASES = {
    "english": "en",
    "italian": "it",
    "japanese": "ja",
    "turkish": "tr",
    "hindi": "hi",
    "indian": "hi",
}
SUPPORTED_TRANSLATION_LANGS = {"en", "it", "hi", "tr", "ja"}
SUBTITLE_LINE_PATTERN = re.compile(r"^(\s*[\d.]+\s*-\s*[\d.]+:\s*)(.*?)(\s*)$")
SUBTITLE_TRANSLATION_BATCH_SIZE = 256
SUBTITLE_TRANSLATION_BATCH_CHARS = 4800
SUBTITLE_TRANSLATION_MARKER_PREFIX = "ZXQ"
SUBTITLE_TRANSLATION_MARKER_SUFFIX = "QXZ"
TRANSLATION_BACKEND = "local"
LOCAL_TRANSLATION_MODEL = "google/madlad400-3b-mt"
LOCAL_TRANSLATION_DEVICE = "auto"
LOCAL_TRANSLATION_BATCH_SIZE = 32
LOCAL_TRANSLATION_MAX_LENGTH = 256
LOCAL_TRANSLATION_NUM_BEAMS = 4
LOCAL_TRANSLATION_REPETITION_PENALTY = 1.2
LOCAL_TRANSLATION_NO_REPEAT_NGRAM_SIZE = 3
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "it": "ita_Latn",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "ja": "jpn_Jpan",
}
MBART50_LANGUAGE_CODES = {
    "en": "en_XX",
    "it": "it_IT",
    "hi": "hi_IN",
    "tr": "tr_TR",
    "ja": "ja_XX",
}
MADLAD_LANGUAGE_CODES = {
    "en": "en",
    "it": "it",
    "hi": "hi",
    "tr": "tr",
    "ja": "ja",
}
SOURCE_TEXT_REPLACEMENTS = {
    "hi": (
        ("खोकला", "खोखला"),
        ("खोखला करता है खोखला करता है एक टाइम का", "समय के साथ खोखला करता है"),
        ("दिस रहा", "दिख रहा"),
        ("साइम", "टाइम"),
        ("खूद", "कूद"),
        ("साफ सुधा", "साफ सुथरा"),
        ("डायबिटी", "डायबिटीज"),
    ),
}
SOURCE_SCRIPT_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "ja": re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]"),
}
_LOCAL_TRANSLATOR_CACHE = {}
_LOCAL_TRANSLATOR_FAILURES = set()
_LOCAL_TRANSLATOR_REPORTED_FAILURES = set()

#CLIENT_SECRETS_FILE = "client_secret.json" #secret json file is needed for using youtube API
SCOPES = [
    'https://www.googleapis.com/auth/youtube',
    'https://www.googleapis.com/auth/youtube.force-ssl'
]

# -----------------------
# Auth & API helpers
# -----------------------

def get_authenticated_service():
    """Authenticate the user and return a YouTube API client."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('youtube', 'v3', credentials=creds)

def extract_playlist_id(playlist_url_or_id: str) -> str:
    """
    Accepts either a full YouTube playlist URL or a bare playlist ID and returns the playlist ID.
    """
    # If it looks like a bare ID, just return it
    if 'http://' not in playlist_url_or_id and 'https://' not in playlist_url_or_id:
        return playlist_url_or_id

    parsed = urlparse(playlist_url_or_id)
    qs = parse_qs(parsed.query)
    playlist_id = qs.get('list', [None])[0]
    if not playlist_id:
        raise ValueError(f"Could not extract playlist ID from: {playlist_url_or_id}")
    return playlist_id

# -----------------------
# Core logic
# -----------------------
# Personalized logger that avoids showing many messages
class YTDummyLogger:
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        # still surface real errors
        print(msg)

# Get the video ids of the playlist
def get_playlist_videos(youtube, playlist_id):
    videos = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        try:
            response = request.execute()
        except HttpError as e:
            print(f"An error occurred: {e}")
            break

        for item in response.get('items', []):
            video_id = item['snippet']['resourceId']['videoId']
            videos.append(video_id)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return videos

# Used to try downloading subs in native language
def get_default_audio_lang(youtube, video_id):
    try:
        resp = youtube.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if items:
            snip = items[0].get("snippet", {})
            return snip.get("defaultAudioLanguage") or snip.get("defaultLanguage")
    except Exception:
        pass
    return None


def subtitle_lang_base(lang_code):
    return str(lang_code).strip().lower().replace("_", "-").split("-")[0]


def normalize_subtitle_lang_suffix(lang_code):
    return str(lang_code or "").strip().lower().replace("_", "-")


def subtitle_path(video_folder, video_id, lang_code):
    return os.path.join(video_folder, f"{video_id}_subtitles_{lang_code}.txt")


def has_exact_subtitles(video_folder, video_id, lang_code):
    return os.path.isfile(subtitle_path(video_folder, video_id, lang_code))


def canonical_subtitle_save_code(lang_code):
    # The dataset loader looks for *_subtitles_en.txt, not en-US/en-GB variants.
    if subtitle_lang_base(lang_code) == "en":
        return "en"
    return lang_code


def normalize_translation_lang_code(lang_code):
    lang = str(lang_code or "").strip().lower().replace("_", "-")
    lang = TRANSLATION_LANGUAGE_ALIASES.get(lang, lang)
    return lang.split("-")[0]


def is_http_429_error(error):
    return "HTTP Error 429" in str(error) or "Too Many Requests" in str(error)


def subtitle_lang_matches(saved_lang, requested_lang):
    saved = normalize_subtitle_lang_suffix(saved_lang)
    requested = normalize_subtitle_lang_suffix(requested_lang)
    if saved == requested:
        return True

    requested_base = subtitle_lang_base(requested)
    if saved == requested_base:
        return True

    # Accept regional variants like en-US/en-GB, but do not treat en_v2 as en.
    if saved.startswith(f"{requested_base}-"):
        region = saved[len(requested_base) + 1:]
        return region.isalpha() and 2 <= len(region) <= 4

    return False


def allows_base_subtitle_fallback(lang_code):
    normalized = normalize_subtitle_lang_suffix(lang_code)
    base = subtitle_lang_base(normalized)
    if normalized == base:
        return True
    if normalized.startswith(f"{base}-"):
        region = normalized[len(base) + 1:]
        return region.isalpha() and 2 <= len(region) <= 4
    return False


def find_existing_subtitle_file(video_folder, video_id, lang_codes):
    """Find an existing subtitle file, accepting regional variants."""
    for lang_code in lang_codes:
        if not lang_code:
            continue
        requested = str(lang_code).strip()
        exact = subtitle_path(video_folder, video_id, requested)
        if os.path.isfile(exact):
            return exact, requested

        base = subtitle_lang_base(lang_code)
        if allows_base_subtitle_fallback(lang_code):
            exact = subtitle_path(video_folder, video_id, base)
            if os.path.isfile(exact):
                return exact, base

        prefix = f"{video_id}_subtitles_"
        if not os.path.isdir(video_folder):
            continue
        for filename in sorted(os.listdir(video_folder)):
            if not filename.startswith(prefix) or not filename.endswith(".txt"):
                continue
            saved_lang = filename[len(prefix):-4]
            if subtitle_lang_matches(saved_lang, lang_code):
                return os.path.join(video_folder, filename), base

    return None, None


def infer_source_subtitle_candidates(native=None, preferred_langs=None):
    candidates = []
    if native:
        candidates.append(native)
    if preferred_langs:
        candidates.extend(preferred_langs)
    candidates.extend(["it", "hi", "tr", "ja"])

    seen = set()
    result = []
    for code in candidates:
        base = normalize_translation_lang_code(code)
        if base and base != "en" and base not in seen:
            seen.add(base)
            result.append(base)
    return result


def infer_playlist_language(playlist_title):
    parts = sanitize_folder_name(playlist_title).split("_")
    if len(parts) >= 2 and parts[-1].lower() == "language":
        return normalize_translation_lang_code(parts[-2])
    return None


def parse_existing_text_subtitles(subtitle_file):
    """Parse existing TXT subtitles into cue-level blocks.

    The downloader TXT format is "start - duration: text", but YouTube/TED
    lines can wrap onto following physical lines. We merge continuation lines
    into the cue text so translation covers the whole subtitle.
    """
    cues = []
    current_prefix = None
    current_text_lines = []

    def flush():
        if current_prefix is None:
            return
        text = " ".join(line.strip() for line in current_text_lines if line.strip())
        cues.append((current_prefix, re.sub(r"\s+", " ", text).strip()))

    with open(subtitle_file, "r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.rstrip("\n")
            match = SUBTITLE_LINE_PATTERN.match(line)
            if match:
                flush()
                current_prefix = match.group(1)
                current_text_lines = [match.group(2).strip()]
            elif current_prefix is not None and line.strip():
                current_text_lines.append(line.strip())

    flush()
    return cues


def iter_translation_batches(texts):
    batch = []
    batch_chars = 0
    for text in texts:
        text_len = len(f"{subtitle_translation_marker(len(batch))} {text}")
        separator_len = 1 if batch else 0
        if batch and (
            len(batch) >= SUBTITLE_TRANSLATION_BATCH_SIZE
            or batch_chars + separator_len + text_len > SUBTITLE_TRANSLATION_BATCH_CHARS
        ):
            yield batch
            batch = []
            batch_chars = 0
            text_len = len(f"{subtitle_translation_marker(0)} {text}")
            separator_len = 0
        batch.append(text)
        batch_chars += separator_len + text_len
    if batch:
        yield batch


def subtitle_translation_marker(index):
    return f"{SUBTITLE_TRANSLATION_MARKER_PREFIX}{index:04d}{SUBTITLE_TRANSLATION_MARKER_SUFFIX}"


def split_packed_subtitle_translation(translated_text, expected_count):
    marker_pattern = re.compile(
        rf"{SUBTITLE_TRANSLATION_MARKER_PREFIX}\s*(\d{{4}})\s*{SUBTITLE_TRANSLATION_MARKER_SUFFIX}",
        flags=re.IGNORECASE,
    )
    matches = list(marker_pattern.finditer(translated_text or ""))
    if len(matches) != expected_count:
        return None

    results = [None] * expected_count
    for idx, match in enumerate(matches):
        try:
            cue_idx = int(match.group(1))
        except Exception:
            return None
        if cue_idx < 0 or cue_idx >= expected_count or results[cue_idx] is not None:
            return None

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(translated_text)
        cue_text = translated_text[start:end]
        cue_text = re.sub(r"\s+", " ", cue_text).strip(" :-\n\t")
        results[cue_idx] = cue_text

    if any(item is None for item in results):
        return None
    return results


def resolve_local_translation_device(torch_module):
    if LOCAL_TRANSLATION_DEVICE != "auto":
        return LOCAL_TRANSLATION_DEVICE
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def local_model_language_codes(model_name):
    normalized_name = str(model_name).lower()
    if "madlad" in normalized_name:
        return MADLAD_LANGUAGE_CODES
    if "mbart" in normalized_name:
        return MBART50_LANGUAGE_CODES
    return NLLB_LANGUAGE_CODES


def local_model_family(model_name):
    normalized_name = str(model_name).lower()
    if "madlad" in normalized_name:
        return "madlad"
    if "mbart" in normalized_name:
        return "mbart"
    return "nllb"


def clean_translated_subtitle_text(text):
    text = html.unescape(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bhuckles\b", "hollows out", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z]+)\s+'\s*(s|re|ve|m|d|ll|t)\b", r"\1'\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"([(\"'])\s+", r"\1", text)
    return text


def normalize_source_subtitle_text(text, source_lang):
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    for old, new in SOURCE_TEXT_REPLACEMENTS.get(source_lang, ()):
        normalized = normalized.replace(old, new)
    return normalized


def is_degenerate_translation(text):
    words = re.findall(r"[A-Za-z][A-Za-z']*", clean_translated_subtitle_text(text).lower())
    if len(words) < 10:
        return False

    repeated_run = 1
    for prev, cur in zip(words, words[1:]):
        if cur == prev:
            repeated_run += 1
            if repeated_run >= 4:
                return True
        else:
            repeated_run = 1

    most_common = max(words.count(word) for word in set(words))
    if most_common >= 8 and most_common / len(words) >= 0.35:
        return True

    return len(words) >= 40 and len(set(words)) <= 5


def looks_untranslated_source_text(text, source_lang):
    pattern = SOURCE_SCRIPT_PATTERNS.get(source_lang)
    if pattern is None:
        return False

    text = str(text or "")
    source_chars = len(pattern.findall(text))
    if source_chars < 12:
        return False

    latin_chars = len(re.findall(r"[A-Za-z]", text))
    if latin_chars >= source_chars:
        return False

    visible_chars = sum(1 for char in text if not char.isspace())
    return visible_chars > 0 and source_chars / visible_chars >= 0.35


class LocalSeq2SeqTranslator:
    supports_batch_translation = True
    strict_no_source_fallback = True

    def __init__(self, source, target="en"):
        source = normalize_translation_lang_code(source)
        target = normalize_translation_lang_code(target)
        language_codes = local_model_language_codes(LOCAL_TRANSLATION_MODEL)
        if source not in language_codes:
            raise ValueError(f"Local translation model does not support source language: {source}")
        if target not in language_codes:
            raise ValueError(f"Local translation model does not support target language: {target}")

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Local translation requires torch, transformers, and sentencepiece. "
                "Install the download requirements or run: "
                "python -m pip install transformers sentencepiece"
            ) from e

        self.torch = torch
        self.source = source
        self.target = target
        self.model_family = local_model_family(LOCAL_TRANSLATION_MODEL)
        self.source_lang = language_codes[source]
        self.target_lang = language_codes[target]
        self.device = resolve_local_translation_device(torch)
        self.batch_size = max(1, int(LOCAL_TRANSLATION_BATCH_SIZE))
        self.max_length = max(16, int(LOCAL_TRANSLATION_MAX_LENGTH))

        cache_key = (LOCAL_TRANSLATION_MODEL, self.device)
        if cache_key in _LOCAL_TRANSLATOR_FAILURES:
            raise RuntimeError(
                f"Local translation model previously failed to load in this run: {LOCAL_TRANSLATION_MODEL}"
            )
        if cache_key not in _LOCAL_TRANSLATOR_CACHE:
            print(f"Loading local translation model: {LOCAL_TRANSLATION_MODEL} on {self.device}")
            model_kwargs = {"use_safetensors": True}
            if self.device.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16
            try:
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_TRANSLATION_MODEL)
                model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_TRANSLATION_MODEL, **model_kwargs)
                model.to(self.device)
                model.eval()
                _LOCAL_TRANSLATOR_CACHE[cache_key] = (tokenizer, model)
            except Exception:
                _LOCAL_TRANSLATOR_FAILURES.add(cache_key)
                raise

        self.tokenizer, self.model = _LOCAL_TRANSLATOR_CACHE[cache_key]
        self.target_token_id = None
        if self.model_family != "madlad":
            self.target_token_id = self._language_token_id(self.target_lang)

    def _clear_cuda_cache(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
            try:
                self.torch.cuda.ipc_collect()
            except Exception:
                pass

    def _language_token_id(self, lang_code):
        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if lang_code_to_id and lang_code in lang_code_to_id:
            return lang_code_to_id[lang_code]

        token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Could not resolve local translation language token: {lang_code}")
        return token_id

    def _translate_chunk(
        self,
        chunk,
        num_beams=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
    ):
        if self.model_family == "madlad":
            chunk = [f"<2{self.target_lang}> {text}" for text in chunk]
        else:
            self.tokenizer.src_lang = self.source_lang

        inputs = self.tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generate_kwargs = {
            "max_new_tokens": self.max_length,
            "num_beams": int(num_beams or LOCAL_TRANSLATION_NUM_BEAMS),
            "repetition_penalty": float(repetition_penalty or LOCAL_TRANSLATION_REPETITION_PENALTY),
            "no_repeat_ngram_size": int(no_repeat_ngram_size or LOCAL_TRANSLATION_NO_REPEAT_NGRAM_SIZE),
        }
        if self.target_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = self.target_token_id

        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **generate_kwargs)
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        finally:
            self._clear_cuda_cache()

    def _translate_many(
        self,
        texts,
        batch_size,
        num_beams=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
    ):
        translated = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            try:
                translated.extend(
                    self._translate_chunk(
                        chunk,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                    )
                )
            except RuntimeError as e:
                message = str(e).lower()
                if "out of memory" in message and batch_size > 1:
                    self._clear_cuda_cache()
                    smaller_batch = max(1, batch_size // 2)
                    print(f"[Warn] Local translation OOM; retrying with batch size {smaller_batch}.")
                    translated.extend(
                        self._translate_many(
                            chunk,
                            smaller_batch,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                        )
                    )
                    continue
                raise
        return translated

    def translate_batch(self, texts):
        texts = [str(text).strip() for text in texts]
        if self.source == self.target:
            return texts
        translated = [
            clean_translated_subtitle_text(item)
            for item in self._translate_many(texts, self.batch_size)
        ]

        for idx, (original, item) in enumerate(zip(texts, translated)):
            if (
                not item
                or is_degenerate_translation(item)
                or looks_untranslated_source_text(item, self.source)
            ):
                print("[Warn] Bad local translation detected; retrying cue with stricter decoding.")
                retry = self._translate_many(
                    [original],
                    1,
                    num_beams=max(4, int(LOCAL_TRANSLATION_NUM_BEAMS)),
                    repetition_penalty=1.6,
                    no_repeat_ngram_size=2,
                )[0]
                retry = clean_translated_subtitle_text(retry)
                if (
                    not retry
                    or is_degenerate_translation(retry)
                    or looks_untranslated_source_text(retry, self.source)
                ):
                    raise RuntimeError(
                        f"Could not produce a reliable English translation for cue: {original[:120]}"
                    )
                translated[idx] = retry

        return [
            item if item and item.strip() else original
            for item, original in zip(translated, texts)
        ]

    def translate(self, text):
        return self.translate_batch([text])[0]


def build_subtitle_translator(source_lang, target_lang="en"):
    source_lang = normalize_translation_lang_code(source_lang)
    target_lang = normalize_translation_lang_code(target_lang)
    if source_lang == target_lang:
        return None

    if TRANSLATION_BACKEND == "local":
        try:
            return LocalSeq2SeqTranslator(source_lang, target_lang)
        except Exception as e:
            failure_key = (LOCAL_TRANSLATION_MODEL, str(e))
            if failure_key not in _LOCAL_TRANSLATOR_REPORTED_FAILURES:
                print(f"Local subtitle translator is unavailable: {e}")
                _LOCAL_TRANSLATOR_REPORTED_FAILURES.add(failure_key)
            return None

    if TRANSLATION_BACKEND == "google":
        if GoogleTranslator is None:
            print("deep_translator is not installed; cannot create English subtitles by translation.")
            return None
        translator_source = source_lang if source_lang in SUPPORTED_TRANSLATION_LANGS else "auto"
        try:
            return GoogleTranslator(source=translator_source, target=target_lang)
        except Exception as e:
            print(f"Failed to initialize Google subtitle translator: {e}")
            return None

    if TRANSLATION_BACKEND == "auto":
        try:
            return LocalSeq2SeqTranslator(source_lang, target_lang)
        except Exception as e:
            failure_key = (LOCAL_TRANSLATION_MODEL, str(e))
            if failure_key not in _LOCAL_TRANSLATOR_REPORTED_FAILURES:
                print(f"Local subtitle translator is unavailable, trying Google fallback: {e}")
                _LOCAL_TRANSLATOR_REPORTED_FAILURES.add(failure_key)
        if GoogleTranslator is None:
            print("deep_translator is not installed; cannot create English subtitles by translation.")
            return None
        translator_source = source_lang if source_lang in SUPPORTED_TRANSLATION_LANGS else "auto"
        try:
            return GoogleTranslator(source=translator_source, target=target_lang)
        except Exception as e:
            print(f"Failed to initialize Google subtitle translator: {e}")
            return None

    print(f"Unsupported translation backend: {TRANSLATION_BACKEND}")
    return None


def translate_text_batch(translator, texts):
    if not texts:
        return []

    if getattr(translator, "supports_batch_translation", False):
        try:
            translated = translator.translate_batch(texts)
            return [
                item if item and str(item).strip() else original
                for item, original in zip(translated, texts)
            ]
        except Exception as e:
            if getattr(translator, "strict_no_source_fallback", False):
                raise
            print(f"[Warn] Local batched subtitle translation failed; falling back per cue: {e}")
            translated = []
            for text in texts:
                try:
                    item = translator.translate(text)
                    translated.append(str(item).strip() if item and str(item).strip() else text)
                except Exception as cue_error:
                    print(f"[Warn] Failed to translate subtitle cue, keeping original text: {cue_error}")
                    translated.append(text)
            return translated

    try:
        packed_text = "\n".join(
            f"{subtitle_translation_marker(idx)} {text}"
            for idx, text in enumerate(texts)
        )
        translated_packed = translator.translate(packed_text)
        translated = split_packed_subtitle_translation(translated_packed, len(texts))
        if translated is not None:
            return [
                item if item and item.strip() else original
                for item, original in zip(translated, texts)
            ]
        print("[Warn] Packed subtitle translation could not be split reliably; falling back per cue.")
    except Exception as e:
        print(f"[Warn] Packed subtitle translation failed; falling back per cue: {e}")

    translated = []
    for text in texts:
        try:
            item = translator.translate(text)
            translated.append(str(item).strip() if item and str(item).strip() else text)
        except Exception as e:
            print(f"[Warn] Failed to translate subtitle cue, keeping original text: {e}")
            translated.append(text)
    return translated


def translate_subtitle_cues(cue_texts, translator):
    cache = {}
    unique_texts = []
    for text in cue_texts:
        text = text if text is not None else ""
        stripped = text.strip()
        if not stripped or stripped in cache:
            continue
        cache[stripped] = None
        unique_texts.append(stripped)

    translated_count = 0
    batch_count = 0
    for batch in iter_translation_batches(unique_texts):
        batch_count += 1
        translated_batch = translate_text_batch(translator, batch)
        for original, translated in zip(batch, translated_batch):
            cache[original] = translated
            translated_count += 1

    if getattr(translator, "supports_batch_translation", False):
        print(f"Translated {translated_count} unique subtitle cues with local batched translation.")
    else:
        print(f"Translated {translated_count} unique subtitle cues in {batch_count} packed request(s).")
    return [cache.get((text or "").strip(), text or "") for text in cue_texts]


def find_existing_output_subtitle_file(video_folder, video_id, output_lang_code):
    if normalize_subtitle_lang_suffix(output_lang_code) == "en":
        return find_existing_subtitle_file(video_folder, video_id, ["en"])[0]

    output_file = subtitle_path(video_folder, video_id, output_lang_code)
    if os.path.isfile(output_file):
        return output_file
    return None


def create_english_subtitles_from_original(
    video_folder,
    video_id,
    source_lang_codes,
    force=False,
    output_lang_code="en",
):
    """Create an English subtitle file by translating an existing original-language subtitle file."""
    existing_output = find_existing_output_subtitle_file(video_folder, video_id, output_lang_code)
    if existing_output and not force:
        print("English subtitles already exist. Skipping subtitle translation fallback.")
        return True

    source_file, source_lang = find_existing_subtitle_file(video_folder, video_id, source_lang_codes)
    if source_file is None:
        print("No original-language subtitle file found for translation fallback.")
        return False

    source_lang = normalize_translation_lang_code(source_lang)
    if source_lang == "en":
        return False

    action = "Retranslating" if existing_output and force else "Creating"
    print(f"{action} English subtitles by translating {source_file} ({source_lang} -> en, backend={TRANSLATION_BACKEND}).")
    translator = build_subtitle_translator(source_lang, "en")
    if translator is None:
        return False

    output_file = subtitle_path(video_folder, video_id, output_lang_code)

    try:
        cues = parse_existing_text_subtitles(source_file)
        if not cues:
            print(f"No parseable subtitle cues found in {source_file}")
            return False

        prefixes = [prefix for prefix, _ in cues]
        cue_texts = [
            normalize_source_subtitle_text(text, source_lang)
            for _, text in cues
        ]
        translated_texts = translate_subtitle_cues(cue_texts, translator)

        with open(output_file, "w", encoding="utf-8") as dst:
            for prefix, translated_text in zip(prefixes, translated_texts):
                dst.write(f"{prefix}{translated_text}\n")
        if existing_output and force:
            print(f"✅ English subtitles overwritten by translation: {output_file}")
        else:
            print(f"✅ English subtitles created by translation: {output_file}")
        return True
    except Exception as e:
        print(f"Failed to create translated English subtitles: {e}")
        try:
            if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
                os.remove(output_file)
        except Exception:
            pass
        return False


def translate_existing_subtitle_tree(
    root_folder,
    force_retranslate_english=False,
    english_output_lang_code="en",
):
    """Create missing English subtitles from already downloaded subtitle files."""
    if not os.path.isdir(root_folder):
        print(f"Subtitle root does not exist: {root_folder}")
        return

    subtitle_file_pattern = re.compile(r"^(.+)_subtitles_([A-Za-z][A-Za-z0-9_-]*)\.txt$")
    created = 0
    overwritten = 0
    already_present = 0
    skipped = 0
    failed = 0

    for video_folder, _, filenames in os.walk(root_folder):
        subtitles_by_video = {}
        for filename in filenames:
            match = subtitle_file_pattern.match(filename)
            if not match:
                continue
            video_id = match.group(1)
            lang = normalize_translation_lang_code(match.group(2))
            subtitles_by_video.setdefault(video_id, set()).add(lang)

        if not subtitles_by_video:
            continue

        playlist_language = infer_playlist_language(os.path.basename(os.path.dirname(video_folder)))
        for video_id, langs in sorted(subtitles_by_video.items()):
            existing_output = find_existing_output_subtitle_file(
                video_folder,
                video_id,
                english_output_lang_code,
            )
            if existing_output and not force_retranslate_english:
                already_present += 1
                continue

            source_candidates = []
            if playlist_language and playlist_language in langs and playlist_language != "en":
                source_candidates.append(playlist_language)
            source_candidates.extend(
                lang for lang in ("it", "hi", "tr", "ja") if lang in langs and lang not in source_candidates
            )
            source_candidates.extend(
                lang for lang in sorted(langs) if lang != "en" and lang not in source_candidates
            )

            if not source_candidates:
                skipped += 1
                continue

            if create_english_subtitles_from_original(
                video_folder,
                video_id,
                source_candidates,
                force=force_retranslate_english,
                output_lang_code=english_output_lang_code,
            ):
                if existing_output:
                    overwritten += 1
                else:
                    created += 1
            else:
                failed += 1

    print(
        "Existing subtitle scan complete: "
        f"created={created}, overwritten={overwritten}, "
        f"already_present={already_present}, skipped={skipped}, failed={failed}"
    )


def parse_subtitle_timestamp(value):
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported subtitle timestamp: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_subtitle_text(lines):
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>", "", text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_downloaded_subtitle_file(subtitle_file):
    rows = []
    with open(subtitle_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == "WEBVTT" or line.startswith(("Kind:", "Language:")):
            i += 1
            continue

        if line.startswith(("NOTE", "STYLE", "REGION")):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        # Cue identifier line in VTT/SRT.
        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        start_raw, end_raw = line.split("-->", 1)
        try:
            start = parse_subtitle_timestamp(start_raw)
            end = parse_subtitle_timestamp(end_raw.strip().split()[0])
        except Exception:
            i += 1
            continue

        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1

        text = clean_subtitle_text(text_lines)
        if text:
            rows.append({
                "start": start,
                "duration": max(0.0, end - start),
                "text": text,
            })

    return rows


def choose_available_subtitle_lang(requested_code, manual_subtitles, automatic_captions):
    requested_base = subtitle_lang_base(requested_code)

    for source in (manual_subtitles, automatic_captions):
        if requested_code in source:
            return requested_code
        for available_code in sorted(source.keys()):
            if subtitle_lang_base(available_code) == requested_base:
                return available_code

    return None


def yt_dlp_subtitle_opts(outtmpl=None, subtitles_langs=None):
    opts = {
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
        "skip_download": True,
        "noplaylist": True,
        "logger": YTDummyLogger(),
    }

    if outtmpl is not None:
        opts["outtmpl"] = outtmpl
    if subtitles_langs is not None:
        opts.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": subtitles_langs,
            "subtitlesformat": "vtt/srt/best",
        })

    if COOKIES_FILE:
        opts["cookiefile"] = COOKIES_FILE
    if COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = COOKIES_FROM_BROWSER
    if YTDLP_PROXY:
        opts["proxy"] = YTDLP_PROXY
    if YTDLP_IMPERSONATE:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        opts["impersonate"] = ImpersonateTarget.from_str(YTDLP_IMPERSONATE.lower())
    if YTDLP_SLEEP_REQUESTS > 0:
        opts["sleep_interval_requests"] = YTDLP_SLEEP_REQUESTS
    if YTDLP_SLEEP_SUBTITLES > 0:
        opts["sleep_interval_subtitles"] = YTDLP_SLEEP_SUBTITLES

    return opts


def download_subtitles_with_ytdlp(video_id, video_folder, language_codes):
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Trying yt-dlp subtitle fallback for: {language_codes}")
    saw_http_429 = False

    try:
        with yt_dlp.YoutubeDL(yt_dlp_subtitle_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        saw_http_429 = is_http_429_error(e)
        print(f"yt-dlp subtitle metadata lookup failed: {e}")
        return False, saw_http_429

    manual_subtitles = info.get("subtitles") or {}
    automatic_captions = info.get("automatic_captions") or {}
    saved_any = False
    attempted_langs = set()
    saved_codes = set()

    for requested_code in language_codes:
        selected_lang = choose_available_subtitle_lang(requested_code, manual_subtitles, automatic_captions)
        if not selected_lang or selected_lang in attempted_langs:
            continue

        attempted_langs.add(selected_lang)
        save_code = canonical_subtitle_save_code(selected_lang)
        if save_code in saved_codes:
            continue
        existing_file, _ = find_existing_subtitle_file(video_folder, video_id, [save_code])
        if existing_file:
            print(f"Subtitles already exist ({save_code}). Skipping yt-dlp download.")
            saved_codes.add(save_code)
            saved_any = True
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, f"{video_id}.%(ext)s")
            try:
                with yt_dlp.YoutubeDL(yt_dlp_subtitle_opts(outtmpl, [selected_lang])) as ydl:
                    ydl.download([url])
            except Exception as e:
                saw_http_429 = saw_http_429 or is_http_429_error(e)
                print(f"yt-dlp failed to download {selected_lang} subtitles: {e}")
                continue

            subtitle_files = []
            for filename in os.listdir(tmpdir):
                path = os.path.join(tmpdir, filename)
                ext = os.path.splitext(filename)[1].lower()
                if os.path.isfile(path) and ext in (".vtt", ".srt"):
                    subtitle_files.append(path)

            if not subtitle_files:
                print(f"yt-dlp did not write a parseable subtitle file for {selected_lang}.")
                continue

            try:
                data = parse_downloaded_subtitle_file(sorted(subtitle_files)[0])
                if not data:
                    print(f"yt-dlp subtitle file for {selected_lang} contained no cues.")
                    continue
                save_subs(video_folder, video_id, save_code, data)
                saved_codes.add(save_code)
                saved_any = True
                print(f"Successfully saved yt-dlp subtitles {selected_lang} as {save_code}")
            except Exception as e:
                print(f"Failed to parse/save yt-dlp subtitles for {selected_lang}: {e}")

    return saved_any, saw_http_429


def download_subtitles(video_id, video_folder, youtube=None, preferred_langs=None, force_english=False,
                       subtitle_backend="auto"):
    """
    Download subtitles using youtube-transcript-api v1.x instance API.
    Tries multiple approaches (preferred langs, direct EN, translation to EN, first-available).

    When force_english=True, always tries to save a direct English transcript as
    *_subtitles_en.txt even if native-language subtitles are already present.
    """
    print(f"Attempting to download subtitles for video: {video_id}")

    preferred_bases = []
    if preferred_langs:
        preferred_bases.extend([
            normalize_translation_lang_code(c)
            for c in preferred_langs
            if normalize_translation_lang_code(c)
        ])

    native = next((c for c in preferred_bases if c != "en"), None)
    if native:
        print(f"Original language from playlist/preference: {native}")

    original_subtitle_candidates = infer_source_subtitle_candidates(native, preferred_bases)
    original_subtitle_file, _ = find_existing_subtitle_file(
        video_folder, video_id, original_subtitle_candidates
    )
    english_subtitle_file, _ = find_existing_subtitle_file(video_folder, video_id, ["en"])

    if original_subtitle_file and english_subtitle_file:
        print("Original and English subtitles already exist. Skipping subtitle services.")
        return

    if not native:
        native = get_default_audio_lang(youtube, video_id) if youtube else None
        if native:
            native = normalize_translation_lang_code(native)
            print(f"Native language detected: {native}")
            original_subtitle_candidates = infer_source_subtitle_candidates(native, preferred_bases)
            original_subtitle_file, _ = find_existing_subtitle_file(
                video_folder, video_id, original_subtitle_candidates
            )
            english_subtitle_file, _ = find_existing_subtitle_file(video_folder, video_id, ["en"])
            if original_subtitle_file and english_subtitle_file:
                print("Original and English subtitles already exist. Skipping subtitle services.")
                return

    # Build language preference list for missing subtitle files only.
    pref_bases = []
    if not original_subtitle_file and native:
        pref_bases.append(native)
    if not english_subtitle_file:
        pref_bases.append("en")

    if not pref_bases:
        print("No missing target subtitles. Skipping subtitle services.")
        return

    if subtitle_backend == "local-translate":
        if english_subtitle_file:
            print("English subtitles already exist. Skipping local translation.")
            return
        if original_subtitle_file:
            create_english_subtitles_from_original(
                video_folder,
                video_id,
                original_subtitle_candidates,
            )
        else:
            print("Local-translate backend requires existing original-language subtitles. Skipping.")
        return

    # Deduplicate while preserving order
    seen = set()
    pref_bases = [x for x in pref_bases if not (x in seen or seen.add(x))]

    # Expand bases into common code variants (e.g., en -> en, en-US, en-GB)
    def expand_codes(code):
        out = []
        code = code.strip()
        out.append(code)  # keep as is
        base = code.split("-")[0].lower()
        out.append(base)  # base-only, lowercase
        # if no region provided, add common regions
        if "-" not in code:
            out.extend([f"{base}-US", f"{base}-GB", f"{base}-{base.upper()}"])
        else:
            # normalize case for region
            parts = code.split("-", 1)
            out.append(f"{parts[0].lower()}-{parts[1].upper()}")
        # dedup preserving order for this single code
        s = set()
        return [c for c in out if not (c in s or s.add(c))]

    expanded_pref = []
    for c in pref_bases:
        for v in expand_codes(c):
            if v not in expanded_pref:
                expanded_pref.append(v)

    print(f"Language preference order (expanded): {expanded_pref}")

    # Keep yt-dlp downloads narrow. YouTube exposes auto-translated subtitles for
    # many languages; downloading all of them quickly triggers HTTP 429.
    ytdlp_pref_bases = list(pref_bases)

    seen = set()
    ytdlp_pref_bases = [x for x in ytdlp_pref_bases if not (x in seen or seen.add(x))]

    ytdlp_pref = []
    for c in ytdlp_pref_bases:
        for v in expand_codes(c):
            if v not in ytdlp_pref:
                ytdlp_pref.append(v)

    print(f"yt-dlp subtitle download order (expanded): {ytdlp_pref}")

    def _save(lang_code, fetched_transcript):
        """Save transcript to disk using the existing save_subs() format."""
        try:
            # fetched_transcript is a FetchedTranscript; convert to raw list[dict]
            data = fetched_transcript.to_raw_data()
            save_subs(video_folder, video_id, lang_code, data)
            return True
        except Exception as e:
            print(f"Failed to save subtitles for {lang_code}: {e}")
            return False

    def _try_ytdlp_fallback():
        if subtitle_backend not in ("auto", "yt-dlp"):
            return False, False
        return download_subtitles_with_ytdlp(video_id, video_folder, ytdlp_pref)

    def _translate_english_from_original_if_needed(reason):
        existing_english, _ = find_existing_subtitle_file(video_folder, video_id, ["en"])
        if existing_english:
            return False
        source_file, _ = find_existing_subtitle_file(
            video_folder, video_id, original_subtitle_candidates
        )
        if not source_file:
            return False
        print(f"{reason}; creating English subtitles from original-language subtitles.")
        return create_english_subtitles_from_original(
            video_folder,
            video_id,
            original_subtitle_candidates,
        )

    def _find_direct_english_transcript(tlist):
        try:
            return tlist.find_transcript(list(ENGLISH_LANGUAGE_CODES))
        except Exception:
            pass

        for transcript in tlist:
            if subtitle_lang_base(transcript.language_code) == "en":
                return transcript
        return None

    def _save_direct_english(tlist):
        tr = _find_direct_english_transcript(tlist)
        if tr is None:
            print("No direct English transcript found.")
            return False

        try:
            fetched = tr.fetch()
            if _save("en", fetched):
                print(f"Successfully saved direct English subtitles ({tr.language_code})")
                return True
        except Exception as e:
            print(f"Failed to save direct English subtitles: {e}")
        return False

    if subtitle_backend == "yt-dlp":
        ytdlp_saved, ytdlp_429 = _try_ytdlp_fallback()
        have_en = find_existing_subtitle_file(video_folder, video_id, ["en"])[0] is not None
        if not have_en and ytdlp_429:
            reason = "HTTP 429 while downloading English subtitles"
        else:
            reason = "English subtitles could not be downloaded"
        if _translate_english_from_original_if_needed(reason):
            ytdlp_saved = True
        if ytdlp_saved:
            return
        print(f"❌ Original and English subtitles are unavailable for video {video_id}. Skipping.")
        return

    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    try:
        ytt = YouTubeTranscriptApi()
        tlist = ytt.list(video_id)

        # Debug: show available tracks
        available = []
        for t in tlist:
            kind = "auto" if t.is_generated else "manual"
            translatable = " (translatable)" if t.is_translatable else ""
            available.append(f"{t.language_code} [{kind}]{translatable}")
        print(f"Available transcripts: {available}")

        saved_any = False

        # 1) Try to save any preferred languages directly (including auto if it matches)
        for code in expanded_pref:
            try:
                tr = tlist.find_transcript([code])
                fetched = tr.fetch()
                # Save using the transcript's language_code (canonical)
                save_code = canonical_subtitle_save_code(tr.language_code)
                if _save(save_code, fetched):
                    saved_any = True
                    print(f"Successfully saved {save_code} subtitles")
            except Exception as e:
                # Not found or failed; continue trying others
                # print(f"No direct transcript for {code}: {e}")
                continue

        # 2) In forced mode, direct English is required even when native subtitles exist.
        english_subtitle_file, _ = find_existing_subtitle_file(video_folder, video_id, ["en"])
        if force_english and not english_subtitle_file:
            print("Force English enabled, attempting direct English subtitles...")
            if _save_direct_english(tlist):
                saved_any = True

        # 3) Try yt-dlp direct subtitles before using transcript translation.
        have_en = find_existing_subtitle_file(video_folder, video_id, ["en"])[0] is not None
        ytdlp_429 = False
        if not have_en:
            ytdlp_saved, ytdlp_429 = _try_ytdlp_fallback()
        else:
            ytdlp_saved = False
        if ytdlp_saved:
            saved_any = True
            have_en = find_existing_subtitle_file(video_folder, video_id, ["en"])[0] is not None
        if not have_en and ytdlp_429:
            reason = "HTTP 429 while downloading English subtitles"
        else:
            reason = "English subtitles could not be downloaded"
        if _translate_english_from_original_if_needed(reason):
            saved_any = True
            have_en = True

        # 4) Ensure English exists: try translate to English if not already saved.
        if force_english and not have_en:
            print("English subtitles are unavailable and no original subtitle file is available for translation.")
        elif not have_en:
            print("No English subtitles found, attempting translation...")
            try:
                # Prefer manual transcripts for translation if available
                translatable_transcripts = sorted(
                    [t for t in tlist if t.is_translatable],
                    key=lambda t: t.is_generated
                )  # manual first (False < True)
                if translatable_transcripts:
                    src = translatable_transcripts[0]
                    en_tr = src.translate("en")
                    fetched_en = en_tr.fetch()
                    if _save("en", fetched_en):
                        saved_any = True
                        print("Successfully saved translated English subtitles")
            except Exception as e:
                print(f"Translation to English failed: {e}")

        # 5) If neither target subtitle could be obtained, skip this video's subtitle update.
        if not saved_any:
            print("No original/English subtitles could be downloaded or created.")

        if saved_any:
            return
        else:
            print(f"❌ No subtitles could be downloaded for video {video_id}")

    except TranscriptsDisabled:
        print(f"Transcripts are disabled for video {video_id}")
        ytdlp_saved, ytdlp_429 = _try_ytdlp_fallback()
        _translate_english_from_original_if_needed("English subtitles could not be downloaded")
        return
    except NoTranscriptFound:
        print(f"No transcripts found for video {video_id}")
        ytdlp_saved, ytdlp_429 = _try_ytdlp_fallback()
        _translate_english_from_original_if_needed("English subtitles could not be downloaded")
        return
    except Exception as e:
        print(f"Transcript API error: {e}")
        ytdlp_saved, ytdlp_429 = _try_ytdlp_fallback()
        _translate_english_from_original_if_needed("English subtitles could not be downloaded")
        return

# Save the subtitles
def save_subs(video_folder, video_id, lang, data):
    """Save subtitle data to file with better error handling"""
    subtitle_file = os.path.join(video_folder, f"{video_id}_subtitles_{lang}.txt")

    # Always try to save (remove the existence check that might prevent re-downloads)
    try:
        with open(subtitle_file, "w", encoding="utf-8") as f:
            for line in data:
                f.write(f"{line['start']} - {line['duration']}: {line['text']}\n")
        print(f"✅ Subtitles saved ({lang}): {subtitle_file}")
    except Exception as e:
        print(f"❌ Failed to save subtitles to {subtitle_file}: {e}")
        raise

# Downloads a video
def download_video(video_id, root_folder, youtube=None, force_english=False, subtitle_backend="auto",
                   preferred_langs=None):
    url = f'https://www.youtube.com/watch?v={video_id}'
    print(f"\n=== Processing video: {url} ===")

    video_folder = os.path.join(root_folder, video_id)
    os.makedirs(video_folder, exist_ok=True)

    base_video = os.path.join(video_folder, f"{video_id}_video.%(ext)s")
    base_audio = os.path.join(video_folder, f"{video_id}_audio.%(ext)s")

    common = {
        "quiet": False,
        "no_warnings": False,
        "noprogress": False,
        "retries": 3,
        "fragment_retries": 3,
        "skip_unavailable_fragments": False,
        "concurrent_fragments": 1,
    }

    video_opts = {
        **common,
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(video_folder, f"{video_id}_video.%(ext)s"),
        "noplaylist": True,
        "continuedl": True,
    }

    audio_opts = {
        **common,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(video_folder, f"{video_id}_audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    if COOKIES_FILE:
        video_opts["cookiefile"] = COOKIES_FILE
        audio_opts["cookiefile"] = COOKIES_FILE

    if COOKIES_FROM_BROWSER:
        # yt-dlp accepts a string like "chrome" or "firefox"
        video_opts["cookiesfrombrowser"] = COOKIES_FROM_BROWSER
        audio_opts["cookiesfrombrowser"] = COOKIES_FROM_BROWSER
    if YTDLP_PROXY:
        video_opts["proxy"] = YTDLP_PROXY
        audio_opts["proxy"] = YTDLP_PROXY
    if YTDLP_IMPERSONATE:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        impersonate_target = ImpersonateTarget.from_str(YTDLP_IMPERSONATE.lower())
        video_opts["impersonate"] = impersonate_target
        audio_opts["impersonate"] = impersonate_target
    if YTDLP_SLEEP_REQUESTS > 0:
        video_opts["sleep_interval_requests"] = YTDLP_SLEEP_REQUESTS
        audio_opts["sleep_interval_requests"] = YTDLP_SLEEP_REQUESTS

    # VIDEO
    try:
        if not any(fn.startswith(f"{video_id}_video.") for fn in os.listdir(video_folder)):
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                ret = ydl.download([url])
            remove_zero_byte_outputs(video_folder, [f"{video_id}_video."])
            if ret != 0:
                raise RuntimeError(...)
        else:
            print("Video already exists. Skipping.")
    except Exception as e:
        print(f"Video download failed ({video_id}). Will still fetch audio/subs. Reason: {e}")

    # AUDIO
    try:
        mp4_path = os.path.join(video_folder, f"{video_id}_video.mp4")
        mp3_path = os.path.join(video_folder, f"{video_id}_audio.mp3")

        if os.path.exists(mp4_path) and (not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0):
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp4_path, "-vn", "-codec:a", "libmp3lame", "-q:a", "2", mp3_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("✅ Extracted MP3 from MP4.")
        else:
            print("Audio already exists (or no MP4 yet). Skipping.")
    except Exception as e:
        print(f"Failed to extract audio for {video_id}: {e}")
    # SUBTITLES (prefer native + English, then fallbacks)
    download_subtitles(
        video_id,
        video_folder,
        youtube=youtube,
        preferred_langs=preferred_langs,
        force_english=force_english,
        subtitle_backend=subtitle_backend,
    )

def sanitize_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

# Process a playlist
def download_playlist(youtube, playlist_url_or_id, root_folder, force_english=False, subtitle_backend="auto"):
    playlist_id = extract_playlist_id(playlist_url_or_id)

    try:
        playlist_request = youtube.playlists().list(
            part="snippet",
            id=playlist_id
        )
        playlist_response = playlist_request.execute()
    except HttpError as e:
        print(f"An error occurred: {e}")
        return

    if not playlist_response.get('items'):
        print(f"No items found for playlist ID: {playlist_id}")
        return

    playlist_title = sanitize_folder_name(playlist_response['items'][0]['snippet']['title'])
    playlist_language = infer_playlist_language(playlist_title)
    print(f"\n==============================")
    print(f"Downloading playlist: {playlist_title} ({playlist_id})")
    if playlist_language:
        print(f"Playlist language: {playlist_language}")
    print(f"==============================")

    video_ids = get_playlist_videos(youtube, playlist_id)

    playlist_folder = os.path.join(root_folder, playlist_title)
    os.makedirs(playlist_folder, exist_ok=True)

    for vid in video_ids:
        download_video(
            vid,
            playlist_folder,
            youtube=youtube,
            force_english=force_english,
            subtitle_backend=subtitle_backend,
            preferred_langs=[playlist_language] if playlist_language else None,
        )

# -----------------------
# CLI
# -----------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Download videos, audios, and subtitles from YouTube playlists."
    )

    p.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Read cookies directly from a browser profile (e.g. 'chrome', 'firefox', 'chromium', 'brave')."
    )

    p.add_argument(
        "-o", "--root-folder",
        required=True,
        help="Root output folder. This path must be supplied explicitly."
    )
    p.add_argument(
        "-p", "--playlists",
        nargs='*',
        default=DEFAULT_PLAYLIST_LINKS,
        help="One or more playlist URLs or IDs. "
             "If omitted, uses the script's default playlist list."
    )
    p.add_argument(
        "--cookies",
        default=None,
        help="Path to a Netscape cookies.txt (recommended in EU)."
    )
    p.add_argument(
        "--force-english",
        action="store_true",
        help=(
            "Always try to download direct English subtitles as *_subtitles_en.txt, "
            "even when original-language subtitles are already present. If YouTube "
            "returns HTTP 429, create English subtitles by translating the original "
            "subtitle file when available."
        )
    )
    p.add_argument(
        "--subtitle-backend",
        choices=("auto", "transcript-api", "yt-dlp", "local-translate"),
        default="auto",
        help=(
            "Subtitle backend. 'auto' tries youtube-transcript-api first and falls back "
            "to yt-dlp. Use 'yt-dlp' when youtube-transcript-api is IP-blocked. "
            "Use 'local-translate' to avoid YouTube caption calls and create missing "
            "English subtitles from already downloaded original-language subtitles."
        )
    )
    p.add_argument(
        "--translation-backend",
        choices=("local", "google", "auto"),
        default="local",
        help=(
            "Backend used when English subtitles must be created from original-language "
            "subtitles. 'local' uses a local seq2seq model and does not call translation APIs."
        )
    )
    p.add_argument(
        "--local-translation-model",
        default=LOCAL_TRANSLATION_MODEL,
        help="Hugging Face seq2seq model used by --translation-backend local."
    )
    p.add_argument(
        "--local-translation-device",
        default="auto",
        help="Device for local translation: auto, cpu, cuda, or cuda:0."
    )
    p.add_argument(
        "--translation-batch-size",
        type=int,
        default=LOCAL_TRANSLATION_BATCH_SIZE,
        help="Number of subtitle cues per local model batch."
    )
    p.add_argument(
        "--translation-max-length",
        type=int,
        default=LOCAL_TRANSLATION_MAX_LENGTH,
        help="Maximum generated tokens per translated subtitle cue."
    )
    p.add_argument(
        "--translate-existing-subtitles-only",
        action="store_true",
        help=(
            "Do not call YouTube or download media. Walk --root-folder and create missing "
            "English subtitle files from already existing original-language subtitle files."
        )
    )
    p.add_argument(
        "--force-retranslate-english",
        action="store_true",
        help=(
            "With --translate-existing-subtitles-only, overwrite existing *_subtitles_en.txt "
            "files by retranslating from the original-language subtitle file."
        )
    )
    p.add_argument(
        "--english-output-lang-code",
        default="en",
        help=(
            "Subtitle language suffix to write when translating existing subtitles. "
            "Use 'en_v2' to create *_subtitles_en_v2.txt without touching *_subtitles_en.txt."
        )
    )
    p.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL passed to yt-dlp, e.g. socks5://user:pass@host:port/."
    )
    p.add_argument(
        "--impersonate",
        default=None,
        help="Browser impersonation target passed to yt-dlp, e.g. chrome or firefox."
    )
    p.add_argument(
        "--sleep-requests",
        type=float,
        default=0.0,
        help="Seconds yt-dlp sleeps between extraction requests."
    )
    p.add_argument(
        "--sleep-subtitles",
        type=float,
        default=0.0,
        help="Seconds yt-dlp sleeps before each subtitle download."
    )
    return p

def remove_zero_byte_outputs(folder, prefixes):
    for fn in os.listdir(folder):
        if any(fn.startswith(p) for p in prefixes):
            path = os.path.join(folder, fn)
            if os.path.isfile(path) and os.path.getsize(path) == 0:
                os.remove(path)
                print(f"Removed empty file: {path}")
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    global COOKIES_FROM_BROWSER
    COOKIES_FROM_BROWSER = args.cookies_from_browser
    global YTDLP_PROXY, YTDLP_IMPERSONATE, YTDLP_SLEEP_REQUESTS, YTDLP_SLEEP_SUBTITLES
    YTDLP_PROXY = args.proxy
    YTDLP_IMPERSONATE = args.impersonate
    YTDLP_SLEEP_REQUESTS = max(0.0, float(args.sleep_requests))
    YTDLP_SLEEP_SUBTITLES = max(0.0, float(args.sleep_subtitles))
    global TRANSLATION_BACKEND, LOCAL_TRANSLATION_MODEL, LOCAL_TRANSLATION_DEVICE
    global LOCAL_TRANSLATION_BATCH_SIZE, LOCAL_TRANSLATION_MAX_LENGTH
    TRANSLATION_BACKEND = args.translation_backend
    LOCAL_TRANSLATION_MODEL = args.local_translation_model
    LOCAL_TRANSLATION_DEVICE = args.local_translation_device
    LOCAL_TRANSLATION_BATCH_SIZE = max(1, int(args.translation_batch_size))
    LOCAL_TRANSLATION_MAX_LENGTH = max(16, int(args.translation_max_length))

    root_folder = args.root_folder
    playlists = args.playlists if args.playlists else DEFAULT_PLAYLIST_LINKS

    os.makedirs(root_folder, exist_ok=True)

    # store globally or pass down; here we store globally for simplicity
    global COOKIES_FILE
    COOKIES_FILE = args.cookies

    if args.translate_existing_subtitles_only:
        translate_existing_subtitle_tree(
            root_folder,
            force_retranslate_english=args.force_retranslate_english,
            english_output_lang_code=args.english_output_lang_code,
        )
        return

    youtube = get_authenticated_service()

    for playlist_link in playlists:
        try:
            download_playlist(
                youtube,
                playlist_link,
                root_folder,
                force_english=args.force_english,
                subtitle_backend=args.subtitle_backend,
            )
        except Exception as e:
            print(f"Failed playlist {playlist_link}: {e}")

if __name__ == '__main__':
    main()
