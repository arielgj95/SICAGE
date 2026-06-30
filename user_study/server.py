#!/usr/bin/env python3
import argparse
import csv
import json
import socket
import threading
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlsplit

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_study.common import CONDITIONS, QUESTION_DEFS, build_study_paths


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest_relative_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _media_url(relpath: str) -> str:
    safe = relpath.replace("\\", "/")
    return f"/media/{safe}"


def _normalize_condition_label(cond: str) -> str:
    mapping = {
        "no_culture": "No Culture",
        "fishr": "Fishr",
        "adversarial": "Adversarial",
        "real": "Real",
    }
    return mapping.get(cond, cond)


def _discover_access_urls(host: str, port: int) -> List[str]:
    if host not in ("0.0.0.0", "::", ""):
        return [f"http://{host}:{port}"]

    urls = [f"http://127.0.0.1:{port}"]
    seen = set()

    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        if ip and not ip.startswith("127."):
            seen.add(ip)
    except Exception:
        pass
    finally:
        if probe is not None:
            probe.close()

    try:
        addr_infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        for info in addr_infos:
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                seen.add(ip)
    except Exception:
        pass

    for ip in sorted(seen):
        urls.append(f"http://{ip}:{port}")
    return urls


class StudyRuntime:
    def __init__(
        self,
        manifest_path: Path,
        mode: str,
        seed: int,
        public_url: str = "",
        access_token: str = "",
        secure_cookies: bool = False,
        trust_forwarded_for: bool = False,
    ):
        self.manifest_path = manifest_path.resolve()
        self.manifest = _read_json(self.manifest_path)
        self.output_root = _resolve_manifest_relative_path(
            self.manifest_path,
            str(self.manifest["output_root"]),
        )
        self.mode = mode
        self.seed = seed
        self.public_url = str(public_url).strip()
        self.access_token = str(access_token).strip()
        self.secure_cookies = bool(secure_cookies)
        self.trust_forwarded_for = bool(trust_forwarded_for)
        self.lock = threading.Lock()
        self.sessions: Dict[str, dict] = {}

        paths = build_study_paths(self.output_root)
        self.results_dir = paths.result_dir
        self.participants_dir = paths.participant_dir
        self.sessions_dir = self.results_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / "all_ratings.csv"

    def _new_session(self) -> dict:
        session_id = uuid.uuid4().hex
        intro_order = [x["intro_id"] for x in self.manifest["intro_videos"]]
        intro_order.sort()

        session = {
            "session_id": session_id,
            "created_at": _utc_now(),
            "participant_started": False,
            "participant": {},
            "intro_order": intro_order,
            "intro_index": 0,
            "trial_order": [],
            "trial_index": 0,
            "responses": [],
            "completed": False,
            "completed_at": None,
        }
        self.sessions[session_id] = session
        self._persist_session(session)
        return session

    def _persist_session(self, session: dict) -> None:
        session_path = self.sessions_dir / f"{session['session_id']}.json"
        _write_json(session_path, session)

    def create_session(self) -> dict:
        with self.lock:
            return self._new_session()

    def _append_csv_rows(self, rows: List[dict]) -> None:
        fieldnames = [
            "timestamp",
            "session_id",
            "participant_id",
            "age",
            "gender",
            "own_culture",
            "cultures_interacted",
            "client_ip",
            "remote_addr",
            "forwarded_for",
            "forwarded_proto",
            "host",
            "user_agent",
            "trial_index",
            "sequence_id",
            "culture",
            "condition",
            "question_id",
            "score",
        ]
        all_rows: List[dict] = []
        for session_path in sorted(self.sessions_dir.glob("*.json")):
            try:
                session_payload = _read_json(session_path)
            except Exception:
                continue
            all_rows.extend(self._session_csv_rows(session_payload))

        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

    def _normalized_responses(self, session: dict) -> List[dict]:
        deduped: Dict[int, dict] = {}
        for response in session.get("responses", []):
            try:
                trial_index = int(response.get("trial_index", -1))
            except Exception:
                continue
            if trial_index < 0:
                continue
            deduped[trial_index] = response
        return [deduped[idx] for idx in sorted(deduped.keys())]

    def _find_response(self, session: dict, trial_index: int) -> dict:
        for response in self._normalized_responses(session):
            if int(response.get("trial_index", -1)) == int(trial_index):
                return response
        return {}

    def _answered_trial_indices(self, session: dict) -> List[int]:
        return [int(response["trial_index"]) for response in self._normalized_responses(session)]

    def _all_trials_answered(self, session: dict) -> bool:
        total = len(session.get("trial_order", []))
        return total > 0 and len(self._answered_trial_indices(session)) >= total

    def _first_unanswered_trial_index(self, session: dict) -> int:
        answered = set(self._answered_trial_indices(session))
        for idx in range(len(session.get("trial_order", []))):
            if idx not in answered:
                return idx
        return max(0, len(session.get("trial_order", [])) - 1)

    def _session_csv_rows(self, session: dict) -> List[dict]:
        if not session.get("participant_started"):
            return []
        participant = session.get("participant", {})
        request_meta = participant.get("request_meta", {})
        csv_rows: List[dict] = []
        for response in self._normalized_responses(session):
            ratings = response.get("ratings", {})
            if not isinstance(ratings, dict):
                continue
            for q in QUESTION_DEFS:
                qid = q["id"]
                if qid not in ratings:
                    continue
                csv_rows.append(
                    {
                        "timestamp": response.get("timestamp", ""),
                        "session_id": session.get("session_id", ""),
                        "participant_id": participant.get("participant_id", "anonymous"),
                        "age": participant.get("age", ""),
                        "gender": participant.get("gender", ""),
                        "own_culture": participant.get("own_culture", ""),
                        "cultures_interacted": ",".join(participant.get("cultures_interacted", [])),
                        "client_ip": request_meta.get("client_ip", ""),
                        "remote_addr": request_meta.get("remote_addr", ""),
                        "forwarded_for": request_meta.get("forwarded_for", ""),
                        "forwarded_proto": request_meta.get("forwarded_proto", ""),
                        "host": request_meta.get("host", ""),
                        "user_agent": request_meta.get("user_agent", ""),
                        "trial_index": response.get("trial_index", ""),
                        "sequence_id": response.get("sequence_id", ""),
                        "culture": response.get("culture", ""),
                        "condition": response.get("condition", ""),
                        "question_id": qid,
                        "score": ratings[qid],
                    }
                )
        return csv_rows

    def _persist_participant(self, session: dict) -> Path:
        participant_id = session["participant"].get("participant_id", "anonymous")
        participant_file = self.participants_dir / f"{participant_id}_{session['session_id']}.json"
        _write_json(participant_file, session)
        return participant_file

    def get_or_create_session(self, session_id: str) -> dict:
        with self.lock:
            if session_id and session_id in self.sessions:
                return self.sessions[session_id]
            if session_id:
                session_path = self.sessions_dir / f"{session_id}.json"
                if session_path.exists():
                    payload = _read_json(session_path)
                    self.sessions[session_id] = payload
                    return payload
            return self._new_session()

    def _build_trial_order(self, session: dict) -> List[dict]:
        import random

        rng = random.Random(f"{self.seed}:{session['session_id']}")
        sequences = list(self.manifest["trial_sequences"])
        rng.shuffle(sequences)

        if self.mode == "all_conditions":
            trials = []
            for seq in sequences:
                for cond in CONDITIONS:
                    trials.append(
                        {
                            "sequence_id": seq["sequence_id"],
                            "culture": seq["culture"],
                            "condition": cond,
                            "condition_label": _normalize_condition_label(cond),
                            "video_relpath": seq["videos_relpath"][cond],
                            "video_url": _media_url(seq["videos_relpath"][cond]),
                        }
                    )
            rng.shuffle(trials)
            return trials

        # Default: balanced single-condition assignment, balanced within each culture.
        # Example: with 8 sequences per culture and 4 conditions, each participant gets
        # exactly 2 sequences per condition for each culture (32 total over 4 cultures).
        by_culture: Dict[str, List[dict]] = {}
        for seq in sequences:
            by_culture.setdefault(seq["culture"], []).append(seq)

        trials = []
        for culture, culture_sequences in by_culture.items():
            rng.shuffle(culture_sequences)

            cond_pool = []
            full_blocks = len(culture_sequences) // len(CONDITIONS)
            rem = len(culture_sequences) % len(CONDITIONS)
            for _ in range(full_blocks):
                cond_pool.extend(CONDITIONS)
            if rem:
                extra = CONDITIONS.copy()
                rng.shuffle(extra)
                cond_pool.extend(extra[:rem])
            rng.shuffle(cond_pool)

            for seq, cond in zip(culture_sequences, cond_pool):
                trials.append(
                    {
                        "sequence_id": seq["sequence_id"],
                        "culture": culture,
                        "condition": cond,
                        "condition_label": _normalize_condition_label(cond),
                        "video_relpath": seq["videos_relpath"][cond],
                        "video_url": _media_url(seq["videos_relpath"][cond]),
                    }
                )
        rng.shuffle(trials)
        return trials

    def start_participant(self, session: dict, payload: dict, request_meta: dict) -> dict:
        with self.lock:
            participant_id = uuid.uuid4().hex[:8].upper()
            age = int(payload["age"])
            gender = str(payload.get("gender", "")).strip()
            own_culture = str(payload["own_culture"]).strip()
            cultures_interacted = payload.get("cultures_interacted", [])
            if not isinstance(cultures_interacted, list):
                raise ValueError("cultures_interacted must be a list")

            session["participant_started"] = True
            session["participant"] = {
                "participant_id": participant_id,
                "age": age,
                "gender": gender,
                "own_culture": own_culture,
                "cultures_interacted": cultures_interacted,
                "request_meta": {
                    "client_ip": request_meta.get("client_ip", ""),
                    "remote_addr": request_meta.get("remote_addr", ""),
                    "forwarded_for": request_meta.get("forwarded_for", ""),
                    "forwarded_proto": request_meta.get("forwarded_proto", ""),
                    "host": request_meta.get("host", ""),
                    "user_agent": request_meta.get("user_agent", ""),
                },
                "started_at": _utc_now(),
            }
            session["trial_order"] = self._build_trial_order(session)
            session["trial_index"] = 0
            session["intro_index"] = 0
            session["responses"] = []
            session["completed"] = False
            session["completed_at"] = None
            self._persist_session(session)
            return session

    def next_intro(self, session: dict) -> dict:
        with self.lock:
            total_intro = len(session["intro_order"])
            if session["intro_index"] < total_intro:
                session["intro_index"] += 1
            self._persist_session(session)
            return session

    def previous_intro(self, session: dict) -> dict:
        with self.lock:
            if session["intro_index"] > 0:
                session["intro_index"] -= 1
            self._persist_session(session)
            return session

    def set_trial_index(self, session: dict, payload: dict) -> dict:
        with self.lock:
            if session["completed"]:
                return session
            if not session["participant_started"]:
                raise ValueError("Participant has not started yet")
            if session["intro_index"] < len(session["intro_order"]):
                raise ValueError("Finish the intro videos before browsing motion samples")
            total = len(session["trial_order"])
            if total <= 0:
                raise ValueError("No trials are available")
            try:
                trial_index = int(payload.get("trial_index", -1))
            except Exception as exc:
                raise ValueError("trial_index must be an integer") from exc
            if trial_index < 0 or trial_index >= total:
                raise ValueError("trial_index out of range")
            session["trial_index"] = trial_index
            self._persist_session(session)
            return session

    def submit_trial(self, session: dict, payload: dict) -> dict:
        ratings = payload.get("ratings", {})
        if not isinstance(ratings, dict):
            raise ValueError("ratings must be an object")

        with self.lock:
            if session["completed"]:
                return session
            if session["trial_index"] >= len(session["trial_order"]):
                raise ValueError("trial_index out of range")

            trial = session["trial_order"][session["trial_index"]]
            row = {
                "timestamp": _utc_now(),
                "trial_index": session["trial_index"],
                "sequence_id": trial["sequence_id"],
                "culture": trial["culture"],
                "condition": trial["condition"],
                "ratings": {},
            }
            for q in QUESTION_DEFS:
                qid = q["id"]
                if qid not in ratings:
                    raise ValueError(f"Missing rating for question: {qid}")
                score = int(ratings[qid])
                if score < 0 or score > 10:
                    raise ValueError(f"Rating out of range for {qid}: {score}")
                row["ratings"][qid] = score

            replaced = False
            for idx, existing in enumerate(session["responses"]):
                if int(existing.get("trial_index", -1)) == int(row["trial_index"]):
                    session["responses"][idx] = row
                    replaced = True
                    break
            if not replaced:
                session["responses"].append(row)

            if self._all_trials_answered(session):
                session["trial_index"] = row["trial_index"]
            elif row["trial_index"] < len(session["trial_order"]) - 1:
                session["trial_index"] = row["trial_index"] + 1
            else:
                session["trial_index"] = self._first_unanswered_trial_index(session)
            self._persist_session(session)
            self._append_csv_rows([])
            return session

    def finish_participant(self, session: dict) -> dict:
        with self.lock:
            if session["completed"]:
                return session
            if not self._all_trials_answered(session):
                raise ValueError("Please answer all motion samples before finishing the study")
            session["completed"] = True
            session["completed_at"] = _utc_now()
            participant_file = self._persist_participant(session)
            session["participant_file"] = str(participant_file)
            self._persist_session(session)
            self._append_csv_rows([])
            return session

    def public_state(self, session: dict) -> dict:
        intro_lookup = {x["intro_id"]: x for x in self.manifest["intro_videos"]}
        intro_item = None
        if session["intro_index"] < len(session["intro_order"]):
            iid = session["intro_order"][session["intro_index"]]
            raw = intro_lookup[iid]
            intro_item = {
                "intro_id": iid,
                "culture": raw["culture"],
                "culture_display": raw["culture_display"],
                "video_url": _media_url(raw["video_relpath"]),
            }

        trial_item = None
        if session["participant_started"] and session["trial_index"] < len(session["trial_order"]):
            trial = session["trial_order"][session["trial_index"]]
            response = self._find_response(session, session["trial_index"])
            culture_display = trial["culture"].replace("_", " ").title()
            for culture in self.manifest.get("cultures", []):
                if culture.get("key") == trial["culture"]:
                    culture_display = culture.get("display_name", culture_display)
                    break
            trial_item = {
                "trial_index": session["trial_index"],
                "sequence_id": trial["sequence_id"],
                "culture": trial["culture"],
                "culture_display": culture_display,
                "condition": trial["condition"],
                "condition_label": trial["condition_label"],
                "video_url": trial["video_url"],
                "ratings": response.get("ratings", {}),
                "is_answered": bool(response),
            }

        answered_trials = self._answered_trial_indices(session)

        return {
            "study": {
                "name": self.manifest.get("study_name", self.output_root.name),
                "mode": self.mode,
                "public_url": self.public_url,
                "is_public_mode": bool(self.public_url),
                "access_token_required": bool(self.access_token),
                "questions": QUESTION_DEFS,
                "conditions": CONDITIONS,
                "cultures": self.manifest.get("cultures", []),
            },
            "session": {
                "session_id": session["session_id"],
                "participant_started": session["participant_started"],
                "participant": session.get("participant", {}),
                "intro_index": session["intro_index"],
                "intro_total": len(session["intro_order"]),
                "intro_item": intro_item,
                "trial_index": session["trial_index"],
                "trial_total": len(session["trial_order"]),
                "answered_trial_indices": answered_trials,
                "answered_trial_total": len(answered_trials),
                "all_trials_answered": self._all_trials_answered(session),
                "trial_item": trial_item,
                "completed": session["completed"],
                "completed_at": session.get("completed_at"),
                "participant_file": session.get("participant_file"),
            },
        }


class StudyRequestHandler(BaseHTTPRequestHandler):
    runtime: StudyRuntime = None

    def log_message(self, fmt: str, *args):
        # Keep terminal output compact.
        return

    def _cookie_attrs(self, max_age: int = 0) -> str:
        attrs = ["Path=/", "HttpOnly"]
        if max_age > 0:
            attrs.append(f"Max-Age={int(max_age)}")
        if self.runtime.secure_cookies:
            attrs.append("Secure")
            attrs.append("SameSite=None")
        else:
            attrs.append("SameSite=Lax")
        return "; ".join(attrs)

    def _set_session_cookie(self, session_id: str):
        self.send_header("Set-Cookie", f"study_session_id={session_id}; {self._cookie_attrs()}")

    def _set_access_cookie(self, token: str):
        self.send_header("Set-Cookie", f"study_access_token={token}; {self._cookie_attrs(max_age=86400)}")

    def _get_cookie_value(self, name: str) -> str:
        cookie = self.headers.get("Cookie", "")
        parts = [x.strip() for x in cookie.split(";") if x.strip()]
        prefix = f"{name}="
        for part in parts:
            if part.startswith(prefix):
                return part.split("=", 1)[1].strip()
        return ""

    def _request_parts(self):
        return urlsplit(self.path)

    def _query_params(self) -> dict:
        return parse_qs(self._request_parts().query, keep_blank_values=False)

    def _query_token(self) -> str:
        values = self._query_params().get("token", [])
        return values[0].strip() if values else ""

    def _request_path(self) -> str:
        return self._request_parts().path or "/"

    def _get_session_id_from_cookie(self) -> str:
        return self._get_cookie_value("study_session_id")

    def _access_cookie_is_valid(self) -> bool:
        if not self.runtime.access_token:
            return True
        return self._get_cookie_value("study_access_token") == self.runtime.access_token

    def _should_set_access_cookie(self) -> bool:
        if not self.runtime.access_token:
            return False
        return self._query_token() == self.runtime.access_token and not self._access_cookie_is_valid()

    def _authorize_request(self) -> bool:
        if not self.runtime.access_token:
            return True
        if self._access_cookie_is_valid():
            return True
        if self._query_token() == self.runtime.access_token:
            return True
        self._text_response("Forbidden: missing or invalid study access token.", status=403)
        return False

    def _request_meta(self) -> dict:
        remote_addr = self.client_address[0] if self.client_address else ""
        forwarded_for = str(self.headers.get("X-Forwarded-For", "")).strip()
        forwarded_proto = ""
        client_ip = remote_addr
        if self.runtime.trust_forwarded_for and forwarded_for:
            client_ip = forwarded_for.split(",", 1)[0].strip() or remote_addr
            forwarded_proto = str(self.headers.get("X-Forwarded-Proto", "")).strip()
        return {
            "client_ip": client_ip,
            "remote_addr": remote_addr,
            "forwarded_for": forwarded_for,
            "forwarded_proto": forwarded_proto,
            "host": str(self.headers.get("Host", "")).strip(),
            "user_agent": str(self.headers.get("User-Agent", "")).strip(),
        }

    def _read_json_body(self) -> dict:
        content_len = int(self.headers.get("Content-Length", "0") or 0)
        if content_len <= 0:
            return {}
        raw = self.rfile.read(content_len)
        return json.loads(raw.decode("utf-8"))

    def _json_response(
        self,
        payload: dict,
        status: int = 200,
        session_id: str = "",
        set_access_cookie: bool = False,
    ):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if session_id:
            self._set_session_cookie(session_id)
        if set_access_cookie and self.runtime.access_token:
            self._set_access_cookie(self.runtime.access_token)
        self.end_headers()
        self.wfile.write(body)

    def _text_response(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        if path.suffix.lower() == ".mp4":
            ctype = "video/mp4"
        elif path.suffix.lower() == ".json":
            ctype = "application/json; charset=utf-8"
        elif path.suffix.lower() == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix.lower() == ".js":
            ctype = "application/javascript; charset=utf-8"
        else:
            ctype = "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_media(self, relpath: str):
        relpath = relpath.lstrip("/")
        relpath = unquote(relpath)
        target = (self.runtime.output_root / relpath).resolve()
        try:
            target.relative_to(self.runtime.output_root)
        except Exception:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        self._serve_file(target)

    def _html_app(self) -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Co-speech Gestures Across Cultures</title>
  <style>
    :root {
      --bg0: #f7f4ed;
      --bg1: #f0e8d8;
      --ink: #0f1a2b;
      --muted: #4d5a72;
      --card: #ffffff;
      --line: #d9d2c5;
      --accent: #d95f02;
      --accent2: #2f6db5;
      --ok: #157145;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI Variable Text", "Gill Sans", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(1200px 700px at -10% -20%, #ffffff 0%, transparent 60%),
        radial-gradient(800px 500px at 120% 10%, #cfe0f2 0%, transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
      min-height: 100vh;
    }
    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      padding: 20px 14px 32px;
    }
    .hero {
      border: 1px solid var(--line);
      background: linear-gradient(160deg, #fff8ec, #ffffff 60%);
      padding: 16px;
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(40, 40, 40, .08);
      animation: rise .35s ease-out both;
    }
    @keyframes rise { from {opacity:.0; transform: translateY(6px)} to {opacity:1; transform:none} }
    h1 {
      margin: 0 0 8px;
      letter-spacing: .3px;
      font-size: 1.55rem;
    }
    .small { color: var(--muted); font-size: .95rem; line-height: 1.45; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    .card {
      grid-column: span 12;
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 6px 24px rgba(25,25,25,.06);
      animation: rise .2s ease-out both;
    }
    .label {
      font-size: .85rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 4px;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      border: 1px solid #c7c0b4;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 1rem;
      background: #fffcf8;
      color: var(--ink);
    }
    .checks { display:flex; flex-wrap:wrap; gap:10px 14px; margin-top:6px; }
    .checks label { font-size:.95rem; color: var(--ink); }
    .actions { display:flex; gap:10px; margin-top:14px; flex-wrap: wrap; }
    .sample-nav { display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 6px; }
    .sample-link {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width: 40px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid #d8d0c3;
      background: #fff;
      color: var(--ink);
      text-decoration: none;
      font-size: .92rem;
      font-weight: 600;
    }
    .sample-link.answered { background: #f2f8f3; border-color: #abd5bf; }
    .sample-link.current { background: #e8f0fb; border-color: #93b7e3; color: #163e73; }
    .sample-link.locked { opacity: .55; }
    button {
      border: 0;
      border-radius: 12px;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), #ea8d23);
      transition: transform .08s ease;
    }
    button.secondary { background: linear-gradient(135deg, var(--accent2), #4a8bd2); }
    button.ok { background: linear-gradient(135deg, var(--ok), #2a9d66); }
    button:active { transform: translateY(1px); }
    .progress {
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px dashed #cfc6b8;
      background: #fdf8ef;
      margin-bottom: 10px;
      font-size: .95rem;
      color: var(--muted);
    }
    video {
      width: 100%;
      border-radius: 12px;
      border: 1px solid #ddd4c7;
      background: #0f1a2b;
      max-height: 76vh;
    }
    .question {
      margin-top: 10px;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid #ece4d8;
      background: #fffcf7;
    }
    .qtitle { font-size: .95rem; margin-bottom: 7px; }
    .slider-row { display:flex; align-items:center; gap:10px; }
    .slider-row input[type=range] { width: 100%; }
    .value-pill {
      min-width: 38px;
      text-align:center;
      border-radius: 999px;
      border: 1px solid #d8d0c3;
      padding: 3px 8px;
      font-weight: 700;
      background: #fff;
    }
    .hidden { display:none !important; }
    .okbox {
      border: 1px solid #abd5bf;
      background: #eefaf2;
      color: #165a35;
      border-radius: 12px;
      padding: 12px;
    }
    @media (max-width: 720px) {
      .wrap { padding: 12px 8px 24px; }
      h1 { font-size: 1.3rem; }
      .card, .hero { padding: 12px; border-radius: 12px; }
      video { max-height: 56vh; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Co-speech Gestures Across Cultures</h1>
      <div class="small">
        This study asks you to evaluate upper-body motion clips. You will first watch a few TED-talk videos
        from four cultures. Focus on the speakers and try to notice cultural differences in how they gesture.
      </div>
      <div class="small" style="margin-top:6px">
        After that, you will rate motion videos for overall quality and cultural appropriateness. Answer each
        statement with a score from 0 to 10, where 0 means you strongly disagree and 10 means you strongly agree.
      </div>
      <div class="small" style="margin-top:6px">
        Please turn the audio on. Using headphones is strongly recommended.
      </div>
    </div>

    <div class="grid">
      <section class="card" id="startCard">
        <div class="label">Before you start</div>
        <div class="small">Provide a few demographic details, then start the study.</div>
        <div class="small" id="questionGuide" style="margin-top:10px"></div>
        <div style="margin-top:10px">
          <div class="label">Age</div>
          <input type="number" id="age" min="1" max="120" required />
        </div>
        <div style="margin-top:10px">
          <div class="label">Gender (optional)</div>
          <select id="gender">
            <option value="">Prefer not to say</option>
            <option value="woman">Woman</option>
            <option value="man">Man</option>
            <option value="non_binary">Non-binary</option>
            <option value="other">Other</option>
          </select>
        </div>
        <div style="margin-top:10px">
          <div class="label">Your culture / background</div>
          <input type="text" id="ownCulture" />
        </div>
        <div style="margin-top:10px">
          <div class="label">Which of these cultures have you already interacted with?</div>
          <div class="checks" id="cultureChecks"></div>
        </div>
        <div class="actions">
          <button id="startBtn">Start Experiment</button>
        </div>
      </section>

      <section class="card hidden" id="introCard">
        <div class="progress">
          <div id="introProgress"></div>
          <div id="introCulture"></div>
        </div>
        <video id="introVideo" controls preload="metadata"></video>
        <div class="small" style="margin-top:8px">You can replay this video using the player controls.</div>
        <div class="actions">
          <button class="secondary" id="prevIntroBtn">Back</button>
          <button class="secondary" id="nextIntroBtn">Next</button>
        </div>
      </section>

      <section class="card hidden" id="trialCard">
        <div class="progress">
          <div id="trialProgress"></div>
          <div id="trialCulture"></div>
        </div>
        <div class="small" id="trialGuide">You can move across the motion samples before finishing the study.</div>
        <div class="sample-nav" id="motionNav"></div>
        <video id="trialVideo" controls preload="metadata"></video>
        <div class="small" style="margin-top:8px">You can replay before submitting your ratings.</div>
        <div id="questionsBox"></div>
        <div class="actions">
          <button class="secondary" id="prevTrialBtn">Back</button>
          <button class="ok" id="submitTrialBtn">Submit Rating</button>
          <button class="secondary hidden" id="finishStudyBtn">Finish Study</button>
        </div>
      </section>

      <section class="card hidden" id="doneCard">
        <div class="okbox">
          <div style="font-weight:800; margin-bottom:8px;">Experiment Completed</div>
          <div id="doneMsg"></div>
        </div>
        <div class="actions">
          <button class="secondary" id="newSessionBtn">Start Another Session</button>
        </div>
      </section>
    </div>
  </div>

<script>
const state = { data: null };
const qsBox = document.getElementById("questionsBox");

function byId(id) { return document.getElementById(id); }

function stopVideo(id, reset=false) {
  const v = byId(id);
  if (!v) return;
  try { v.pause(); } catch (_) {}
  if (reset) {
    try { v.currentTime = 0; } catch (_) {}
  }
}

function showCard(id) {
  ["introVideo", "trialVideo"].forEach(videoId => {
    stopVideo(videoId, false);
  });
  ["startCard", "introCard", "trialCard", "doneCard"].forEach(x => byId(x).classList.add("hidden"));
  byId(id).classList.remove("hidden");
}

async function api(path, method="GET", body=null) {
  const res = await fetch(path, {
    method,
    headers: body ? {"Content-Type":"application/json"} : {},
    body: body ? JSON.stringify(body) : null
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || ("HTTP " + res.status));
  }
  return await res.json();
}

function renderStart(data) {
  showCard("startCard");
  const guide = byId("questionGuide");
  guide.innerHTML =
    "<strong>You will rate each motion clip using these statements:</strong><br>" +
    data.study.questions.map((q, i) => `${i + 1}. ${q.prompt}`).join("<br>");
  const checks = byId("cultureChecks");
  checks.innerHTML = "";
  data.study.cultures.forEach(c => {
    const id = "chk_" + c.key;
    const wrap = document.createElement("label");
    wrap.innerHTML = `<input type="checkbox" id="${id}" value="${c.key}" /> ${c.display_name}`;
    checks.appendChild(wrap);
  });
}

function renderIntro(data) {
  showCard("introCard");
  const s = data.session;
  byId("introProgress").textContent = `Intro ${s.intro_index + 1} / ${s.intro_total}`;
  byId("introCulture").textContent = s.intro_item ? s.intro_item.culture_display : "";
  byId("prevIntroBtn").disabled = s.intro_index <= 0;
  const v = byId("introVideo");
  stopVideo("trialVideo", true);
  stopVideo("introVideo", true);
  v.src = s.intro_item.video_url;
  v.load();
  byId("nextIntroBtn").textContent = (s.intro_index + 1 >= s.intro_total) ? "Start Rating Phase" : "Next";
}

function sliderHtml(qid, prompt, value) {
  return `
    <div class="question">
      <div class="qtitle">${prompt}</div>
      <div class="slider-row">
        <span style="font-size:.85rem;color:#60708a">0</span>
        <input type="range" min="0" max="10" step="1" value="${value}" id="sl_${qid}" />
        <span style="font-size:.85rem;color:#60708a">10</span>
        <div class="value-pill" id="val_${qid}">${value}</div>
      </div>
    </div>
  `;
}

function renderSampleNav(session) {
  const answered = new Set(session.answered_trial_indices || []);
  const nav = byId("motionNav");
  let html = "";
  for (let i = 0; i < session.trial_total; i += 1) {
    const classes = ["sample-link"];
    if (answered.has(i)) classes.push("answered");
    if (i === session.trial_index) classes.push("current");
    html += `<a href="#" class="${classes.join(" ")}" data-trial-index="${i}">${i + 1}</a>`;
  }
  nav.innerHTML = html;
  nav.querySelectorAll("[data-trial-index]").forEach(el => {
    el.addEventListener("click", async (event) => {
      event.preventDefault();
      const idx = Number(el.getAttribute("data-trial-index"));
      try {
        const data = await api("/api/set_trial", "POST", {trial_index: idx});
        render(data);
      } catch (e) {
        alert(e.message || String(e));
      }
    });
  });
}

function renderTrial(data) {
  showCard("trialCard");
  const s = data.session;
  const currentRatings = (s.trial_item && s.trial_item.ratings) ? s.trial_item.ratings : {};
  byId("trialProgress").textContent = `Trial ${s.trial_index + 1} / ${s.trial_total}`;
  byId("trialCulture").textContent = `Target culture: ${s.trial_item.culture_display}`;
  byId("trialGuide").textContent =
    `Answered ${s.answered_trial_total} of ${s.trial_total}. You can use the numbered links to revisit any motion sample before finishing.`;
  byId("prevTrialBtn").disabled = s.trial_index <= 0;
  renderSampleNav(s);
  const v = byId("trialVideo");
  stopVideo("introVideo", true);
  stopVideo("trialVideo", true);
  v.src = s.trial_item.video_url;
  v.load();
  qsBox.innerHTML = data.study.questions.map(q => {
    const value = Number(currentRatings[q.id]);
    return sliderHtml(q.id, q.prompt, Number.isFinite(value) ? value : 5);
  }).join("");
  data.study.questions.forEach(q => {
    const sl = byId("sl_" + q.id);
    const val = byId("val_" + q.id);
    sl.addEventListener("input", () => { val.textContent = sl.value; });
  });
  const finishBtn = byId("finishStudyBtn");
  finishBtn.classList.toggle("hidden", !s.all_trials_answered);
}

function renderDone(data) {
  showCard("doneCard");
  stopVideo("introVideo", true);
  stopVideo("trialVideo", true);
  const p = data.session.participant || {};
  byId("doneMsg").textContent =
    `Your responses were saved on the study laptop. Your participant code is ${p.participant_id || "pending"}.`;
}

function render(data) {
  state.data = data;
  if (!data.session.participant_started) {
    return renderStart(data);
  }
  if (data.session.completed) {
    return renderDone(data);
  }
  if (data.session.intro_index < data.session.intro_total) {
    return renderIntro(data);
  }
  return renderTrial(data);
}

async function refresh() {
  const data = await api("/api/bootstrap");
  render(data);
}

byId("startBtn").addEventListener("click", async () => {
  try {
    const age = Number(byId("age").value);
    const ownCulture = byId("ownCulture").value.trim();
    if (!age || age < 1 || age > 120) throw new Error("Please provide a valid age.");
    if (!ownCulture) throw new Error("Please provide your culture/background.");
    const culturesInteracted = [];
    document.querySelectorAll("#cultureChecks input[type=checkbox]").forEach(chk => {
      if (chk.checked) culturesInteracted.push(chk.value);
    });
    await api("/api/start", "POST", {
      age,
      gender: byId("gender").value,
      own_culture: ownCulture,
      cultures_interacted: culturesInteracted
    });
    await refresh();
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("nextIntroBtn").addEventListener("click", async () => {
  try {
    await api("/api/next_intro", "POST", {});
    await refresh();
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("prevIntroBtn").addEventListener("click", async () => {
  try {
    await api("/api/previous_intro", "POST", {});
    await refresh();
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("prevTrialBtn").addEventListener("click", async () => {
  try {
    const data = state.data;
    const prevIndex = Math.max(0, data.session.trial_index - 1);
    const nextData = await api("/api/set_trial", "POST", {trial_index: prevIndex});
    render(nextData);
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("submitTrialBtn").addEventListener("click", async () => {
  try {
    const data = state.data;
    const ratings = {};
    data.study.questions.forEach(q => {
      ratings[q.id] = Number(byId("sl_" + q.id).value);
    });
    await api("/api/submit_trial", "POST", {ratings});
    await refresh();
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("finishStudyBtn").addEventListener("click", async () => {
  try {
    await api("/api/finish", "POST", {});
    await refresh();
  } catch (e) {
    alert(e.message || String(e));
  }
});

byId("newSessionBtn").addEventListener("click", async () => {
  try {
    const data = await api("/api/new_session", "POST", {});
    render(data);
    byId("age").value = "";
    byId("gender").value = "";
    byId("ownCulture").value = "";
    document.querySelectorAll("#cultureChecks input[type=checkbox]").forEach(chk => {
      chk.checked = false;
    });
  } catch (e) {
    alert(e.message || String(e));
  }
});

refresh().catch(err => {
  document.body.innerHTML = "<pre style='padding:16px'>" + String(err) + "</pre>";
});
</script>
</body>
</html>"""

    def do_GET(self):
        if not self._authorize_request():
            return
        session_id = self._get_session_id_from_cookie()
        session = self.runtime.get_or_create_session(session_id)
        path = self._request_path()

        if path == "/":
            html = self._html_app()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode("utf-8"))))
            self._set_session_cookie(session["session_id"])
            if self._should_set_access_cookie():
                self._set_access_cookie(self.runtime.access_token)
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if path == "/api/bootstrap":
            payload = self.runtime.public_state(session)
            self._json_response(
                payload,
                session_id=session["session_id"],
                set_access_cookie=self._should_set_access_cookie(),
            )
            return

        if path.startswith("/media/"):
            relpath = path[len("/media/") :]
            self._serve_media(relpath)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self):
        if not self._authorize_request():
            return
        session_id = self._get_session_id_from_cookie()
        session = self.runtime.get_or_create_session(session_id)
        path = self._request_path()

        try:
            payload = self._read_json_body()
            if path == "/api/new_session":
                session = self.runtime.create_session()
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/start":
                session = self.runtime.start_participant(session, payload, self._request_meta())
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/next_intro":
                session = self.runtime.next_intro(session)
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/previous_intro":
                session = self.runtime.previous_intro(session)
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/set_trial":
                session = self.runtime.set_trial_index(session, payload)
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/submit_trial":
                session = self.runtime.submit_trial(session, payload)
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            if path == "/api/finish":
                session = self.runtime.finish_participant(session)
                self._json_response(
                    self.runtime.public_state(session),
                    session_id=session["session_id"],
                    set_access_cookie=self._should_set_access_cookie(),
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        except Exception as exc:
            self._text_response(f"Request error: {exc}", status=400)


def main():
    parser = argparse.ArgumentParser(description="Run local website for the gesture user study.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to study_manifest.json")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--public-url",
        type=str,
        default="",
        help="Optional public HTTPS URL when the server is exposed through a tunnel or reverse proxy.",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default="",
        help="Optional token required to open the study publicly. Share the URL with ?token=<value>.",
    )
    parser.add_argument(
        "--secure-cookies",
        action="store_true",
        help="Mark cookies as Secure and SameSite=None. Use this behind HTTPS/public tunnels.",
    )
    parser.add_argument(
        "--trust-forwarded-for",
        action="store_true",
        help="Trust X-Forwarded-For / X-Forwarded-Proto from a reverse proxy or tunnel.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["balanced_single_condition", "all_conditions"],
        default="balanced_single_condition",
        help="Trial presentation mode.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    runtime = StudyRuntime(
        manifest_path=Path(args.manifest),
        mode=args.mode,
        seed=args.seed,
        public_url=args.public_url,
        access_token=args.access_token,
        secure_cookies=args.secure_cookies,
        trust_forwarded_for=args.trust_forwarded_for,
    )
    StudyRequestHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), StudyRequestHandler)
    urls = _discover_access_urls(args.host, args.port)
    print("Study server running. Open one of these URLs:")
    for url in urls:
        print(f"  - {url}")
    if args.public_url:
        public_url = args.public_url.rstrip("/")
        print(f"Public URL: {public_url}")
        if args.access_token:
            print(f"Share link: {public_url}/?token={args.access_token}")
        else:
            print(f"Share link: {public_url}/")
    elif args.access_token:
        print("Access token enabled. Use ?token=<your token> on the first visit.")
    print(f"Manifest: {runtime.manifest_path}")
    print(f"Results folder: {runtime.results_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
