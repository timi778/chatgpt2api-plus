from __future__ import annotations

import atexit
import copy
import re
import threading
from datetime import datetime, timedelta
from typing import Any

from services.config import DATA_DIR
from services.json_file import read_json_object, write_json_file
from utils.log import logger
from utils.timezone import beijing_now, parse_to_beijing_naive


DASHBOARD_METRICS_FILE = DATA_DIR / "dashboard_metrics.json"
DASHBOARD_METRICS_RETENTION_DAYS = 30
DASHBOARD_METRICS_FLUSH_DELAY_SECONDS = 2.0

_NON_MODEL_KEYS = {
    "",
    "-",
    "auto",
    "default",
    "unknown",
    "null",
    "none",
    "low",
    "medium",
    "high",
    "standard",
    "hd",
    "portrait",
    "landscape",
    "square",
    "vertical",
    "horizontal",
    "image",
    "images",
    "text",
    "chat",
    "generation",
    "generations",
    "edit",
    "edits",
}


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _detail_value(item: dict[str, Any], key: str, default: object = "") -> object:
    detail = item.get("detail")
    if isinstance(detail, dict):
        value = detail.get(key)
        if value not in (None, ""):
            return value
    value = item.get(key)
    return default if value in (None, "") else value


def _parse_log_time(value: object) -> datetime | None:
    return parse_to_beijing_naive(value)


def _beijing_now_naive() -> datetime:
    return beijing_now().replace(tzinfo=None)


def _call_started_at(item: dict[str, Any]) -> object:
    return _detail_value(item, "started_at", item.get("time"))


def _looks_like_model_label(value: object) -> bool:
    label = _clean_text(value)
    key = label.lower().replace("\u00d7", "x")
    if key in _NON_MODEL_KEYS or key.startswith("/"):
        return False
    if re.fullmatch(r"\d+\s*x\s*\d+", key) or re.fullmatch(r"\d+\s*:\s*\d+", key):
        return False
    return bool(label)


def _increment(counter: dict[str, int], key: object, default: str = "unknown") -> None:
    label = _clean_text(key) or default
    counter[label] = int(counter.get(label, 0) or 0) + 1


def _empty_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "success": 0,
        "failed": 0,
        "rate_limited": 0,
        "by_endpoint": {},
        "by_model": {},
        "by_status": {},
        "by_error_code": {},
        "model_total_times": {},
        "model_time_counts": {},
    }


def _merge_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("total", "success", "failed", "rate_limited"):
        target[key] = int(target.get(key, 0) or 0) + int(source.get(key, 0) or 0)
    for key in ("by_endpoint", "by_model", "by_status", "by_error_code", "model_total_times", "model_time_counts"):
        target_map = target.setdefault(key, {})
        source_map = source.get(key) if isinstance(source.get(key), dict) else {}
        for name, value in source_map.items():
            try:
                numeric = float(value) if key == "model_total_times" else int(value)
            except (TypeError, ValueError):
                numeric = 0.0 if key == "model_total_times" else 0
            target_map[str(name)] = target_map.get(str(name), 0) + numeric


class DashboardMetricsService:
    """Rolling dashboard aggregates independent from call-log retention."""

    def __init__(self, path=DASHBOARD_METRICS_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._data: dict[str, Any] | None = None
        self._dirty = False
        self._flush_timer: threading.Timer | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            data = read_json_object(self.path, name="dashboard_metrics.json")
            if not isinstance(data.get("days"), dict):
                data["days"] = {}
            data["version"] = 1
            self._data = data
        return self._data

    def _save(self, data: dict[str, Any]) -> None:
        data["version"] = 1
        data["retention_days"] = DASHBOARD_METRICS_RETENTION_DAYS
        data["updated_at"] = beijing_now().isoformat(timespec="seconds")
        write_json_file(self.path, data)

    @staticmethod
    def _prune(data: dict[str, Any], now: datetime | None = None) -> bool:
        current = (now or _beijing_now_naive()).date()
        cutoff = current - timedelta(days=DASHBOARD_METRICS_RETENTION_DAYS - 1)
        days = data.get("days") if isinstance(data.get("days"), dict) else {}
        changed = False
        for day in list(days.keys()):
            try:
                parsed = datetime.strptime(str(day), "%Y-%m-%d").date()
            except ValueError:
                days.pop(day, None)
                changed = True
                continue
            if parsed < cutoff or parsed > current:
                days.pop(day, None)
                changed = True
        return changed

    def _schedule_flush_locked(self) -> None:
        if self._flush_timer is not None and self._flush_timer.is_alive():
            return
        timer = threading.Timer(DASHBOARD_METRICS_FLUSH_DELAY_SECONDS, self._flush_due)
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _mark_dirty_locked(self) -> None:
        self._dirty = True
        self._schedule_flush_locked()

    def _flush_due(self) -> None:
        try:
            self.flush()
        except Exception as exc:
            logger.error({"event": "dashboard_metrics_flush_failed", "error": str(exc)})

    def flush(self) -> None:
        with self._lock:
            timer = self._flush_timer
            self._flush_timer = None
            if timer is not None and timer is not threading.current_thread():
                timer.cancel()
            if self._data is None or not self._dirty:
                return
            self._prune(self._data)
            try:
                self._save(self._data)
            except Exception:
                self._dirty = True
                self._schedule_flush_locked()
                raise
            else:
                self._dirty = False

    @staticmethod
    def _apply_call(bucket: dict[str, Any], item: dict[str, Any]) -> None:
        status = _clean_text(_detail_value(item, "status", item.get("status"))).lower()
        endpoint = _clean_text(_detail_value(item, "endpoint"))
        model = _clean_text(_detail_value(item, "model"))
        error_code = _clean_text(_detail_value(item, "error_code"))
        is_failed = status in {"failed", "error", "fail"} or bool(_detail_value(item, "error"))

        bucket["total"] = int(bucket.get("total", 0) or 0) + 1
        if is_failed:
            bucket["failed"] = int(bucket.get("failed", 0) or 0) + 1
            if error_code in {"rate_limited", "rate_limit", "429"}:
                bucket["rate_limited"] = int(bucket.get("rate_limited", 0) or 0) + 1
        else:
            bucket["success"] = int(bucket.get("success", 0) or 0) + 1

        if status:
            _increment(bucket.setdefault("by_status", {}), status)
        if endpoint.startswith("/"):
            _increment(bucket.setdefault("by_endpoint", {}), endpoint)
        if _looks_like_model_label(model):
            _increment(bucket.setdefault("by_model", {}), model)
            try:
                duration_ms = max(0.0, float(_detail_value(item, "duration_ms", 0)))
            except (TypeError, ValueError):
                duration_ms = None
            if duration_ms is not None:
                totals = bucket.setdefault("model_total_times", {})
                counts = bucket.setdefault("model_time_counts", {})
                totals[model] = float(totals.get(model, 0.0) or 0.0) + duration_ms
                counts[model] = int(counts.get(model, 0) or 0) + 1
        if error_code:
            _increment(bucket.setdefault("by_error_code", {}), error_code)

    def record_call_log(self, item: dict[str, Any]) -> None:
        dt = _parse_log_time(_call_started_at(item))
        if dt is None:
            return
        with self._lock:
            data = self._load()
            self._prune(data)
            days = data.setdefault("days", {})
            day_key = dt.strftime("%Y-%m-%d")
            hour_key = dt.strftime("%H")
            day = days.setdefault(day_key, _empty_bucket())
            hours = day.setdefault("hours", {})
            hour = hours.setdefault(hour_key, _empty_bucket())
            self._apply_call(day, item)
            self._apply_call(hour, item)
            self._mark_dirty_locked()

    def backfill_if_empty(self, items: list[dict[str, Any]]) -> None:
        with self._lock:
            data = self._load()
            days = data.get("days") if isinstance(data.get("days"), dict) else {}
            if days:
                return
            self._prune(data)
            for item in reversed(items):
                if not isinstance(item, dict):
                    continue
                dt = _parse_log_time(_call_started_at(item))
                if dt is None:
                    continue
                day_key = dt.strftime("%Y-%m-%d")
                hour_key = dt.strftime("%H")
                days = data.setdefault("days", {})
                day = days.setdefault(day_key, _empty_bucket())
                hours = day.setdefault("hours", {})
                hour = hours.setdefault(hour_key, _empty_bucket())
                self._apply_call(day, item)
                self._apply_call(hour, item)
            self._prune(data)
            self._mark_dirty_locked()

    def summary(self, time_range: str = "24h") -> dict[str, Any]:
        bucket_count = {"24h": 24, "7d": 7, "30d": 30}.get(time_range, 24)
        bucket_delta = timedelta(hours=1) if time_range == "24h" else timedelta(days=1)
        bucket_format = "%H:00" if time_range == "24h" else "%m-%d"
        raw_now = _beijing_now_naive()
        current_bucket_start = (
            raw_now.replace(minute=0, second=0, microsecond=0)
            if time_range == "24h"
            else raw_now.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        starts = [current_bucket_start - bucket_delta * (bucket_count - 1 - index) for index in range(bucket_count)]
        labels = [start.strftime(bucket_format) for start in starts]

        with self._lock:
            data = self._load()
            if self._prune(data):
                self._mark_dirty_locked()
            raw_days = data.get("days") if isinstance(data.get("days"), dict) else {}
            days = copy.deepcopy(raw_days)

        total_bucket = _empty_bucket()
        series_buckets: list[dict[str, Any]] = []
        for start in starts:
            if time_range == "24h":
                day = days.get(start.strftime("%Y-%m-%d"), {})
                hours = day.get("hours") if isinstance(day, dict) and isinstance(day.get("hours"), dict) else {}
                bucket = hours.get(start.strftime("%H"), {}) if isinstance(hours, dict) else {}
            else:
                bucket = days.get(start.strftime("%Y-%m-%d"), {})
            safe_bucket = bucket if isinstance(bucket, dict) else {}
            series_buckets.append(safe_bucket)
            _merge_bucket(total_bucket, safe_bucket)

        model_requests: dict[str, list[int]] = {}
        model_total_times: dict[str, list[float]] = {}
        model_time_counts: dict[str, list[int]] = {}
        for index, bucket in enumerate(series_buckets):
            by_model = bucket.get("by_model") if isinstance(bucket.get("by_model"), dict) else {}
            total_times = bucket.get("model_total_times") if isinstance(bucket.get("model_total_times"), dict) else {}
            time_counts = bucket.get("model_time_counts") if isinstance(bucket.get("model_time_counts"), dict) else {}
            for model, count in by_model.items():
                model_requests.setdefault(str(model), [0] * bucket_count)[index] += int(count or 0)
            for model, total in total_times.items():
                model_total_times.setdefault(str(model), [0.0] * bucket_count)[index] += float(total or 0.0)
            for model, count in time_counts.items():
                model_time_counts.setdefault(str(model), [0] * bucket_count)[index] += int(count or 0)

        model_avg_times = {
            model: [
                round(totals[index] / counts[index], 2) if counts[index] > 0 else 0.0
                for index in range(bucket_count)
            ]
            for model, totals in model_total_times.items()
            for counts in [model_time_counts.get(model, [0] * bucket_count)]
        }

        failed_requests = [
            max(0, int(bucket.get("failed", 0) or 0) - int(bucket.get("rate_limited", 0) or 0))
            for bucket in series_buckets
        ]
        return {
            "total": int(total_bucket.get("total", 0) or 0),
            "success": int(total_bucket.get("success", 0) or 0),
            "failed": int(total_bucket.get("failed", 0) or 0),
            "by_endpoint": total_bucket.get("by_endpoint", {}),
            "by_model": total_bucket.get("by_model", {}),
            "by_status": total_bucket.get("by_status", {}),
            "by_error_code": total_bucket.get("by_error_code", {}),
            "recent_failures": [],
            "source": "dashboard_metrics",
            "retention_days": DASHBOARD_METRICS_RETENTION_DAYS,
            "trend": {
                "labels": labels,
                "total_requests": [int(bucket.get("total", 0) or 0) for bucket in series_buckets],
                "success_requests": [int(bucket.get("success", 0) or 0) for bucket in series_buckets],
                "failed_requests": failed_requests,
                "rate_limited_requests": [int(bucket.get("rate_limited", 0) or 0) for bucket in series_buckets],
                "model_requests": model_requests,
                "model_ttfb_times": {},
                "model_total_times": model_avg_times,
            },
        }


dashboard_metrics_service = DashboardMetricsService()
atexit.register(dashboard_metrics_service.flush)


def safe_record_dashboard_call(item: dict[str, Any]) -> None:
    try:
        dashboard_metrics_service.record_call_log(item)
    except Exception as exc:
        logger.error({"event": "dashboard_metrics_record_failed", "error": str(exc)})
