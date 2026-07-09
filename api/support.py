from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

from fastapi import HTTPException, Request

from services.account_service import account_service
from services.auth_service import auth_service
from services.config import config

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIST_DIR = BASE_DIR / "web_dist"
ACCOUNT_AUTO_REFRESH_BATCH_SIZE = 5


def extract_bearer_token(authorization: str | None) -> str:
    scheme, _, value = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return ""
    return value.strip()


def _legacy_admin_identity(token: str) -> dict[str, object] | None:
    auth_key = str(config.auth_key or "").strip()
    if auth_key and token == auth_key:
        return {"id": "admin", "name": "管理员", "role": "admin"}
    return None


def require_identity(authorization: str | None) -> dict[str, object]:
    token = extract_bearer_token(authorization)
    identity = _legacy_admin_identity(token) or auth_service.authenticate(token)
    if identity is None:
        raise HTTPException(status_code=401, detail={"error": "密钥无效或已失效，请重新登录"})
    return identity


def require_auth_key(authorization: str | None) -> None:
    require_identity(authorization)


def require_admin(authorization: str | None) -> dict[str, object]:
    identity = require_identity(authorization)
    if identity.get("role") != "admin":
        raise HTTPException(status_code=403, detail={"error": "需要管理员权限才能执行这个操作"})
    return identity


def resolve_image_base_url(request: Request) -> str:
    return config.base_url or f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"


def raise_image_quota_error(exc: Exception) -> None:
    message = str(exc)
    status_code = int(getattr(exc, "status_code", 0) or 0)
    code = str(getattr(exc, "code", "") or getattr(exc, "kind", "") or "").strip()
    if status_code:
        raise HTTPException(status_code=status_code, detail={"error": code or message}) from exc
    lower = message.lower()
    if "image_account_selection:quota_exhausted" in lower or "insufficient_quota" in lower:
        raise HTTPException(status_code=429, detail={"error": "insufficient_quota"}) from exc
    if "image_account_selection:" in lower or "no available image quota" in lower:
        raise HTTPException(status_code=503, detail={"error": message}) from exc
    raise HTTPException(status_code=502, detail={"error": message}) from exc


def sanitize_cpa_pool(pool: dict | None) -> dict | None:
    if not isinstance(pool, dict):
        return None
    return {key: value for key, value in pool.items() if key != "secret_key"}


def sanitize_cpa_pools(pools: list[dict]) -> list[dict]:
    return [sanitized for pool in pools if (sanitized := sanitize_cpa_pool(pool)) is not None]


def sanitize_sub2api_server(server: dict | None) -> dict | None:
    if not isinstance(server, dict):
        return None
    sanitized = {key: value for key, value in server.items() if key not in {"password", "api_key"}}
    sanitized["has_api_key"] = bool(str(server.get("api_key") or "").strip())
    return sanitized


def sanitize_sub2api_servers(servers: list[dict]) -> list[dict]:
    return [sanitized for server in servers if (sanitized := sanitize_sub2api_server(server)) is not None]


def _account_refresh_interval_seconds() -> int:
    try:
        minutes = int(config.refresh_account_interval_minute)
    except (TypeError, ValueError):
        minutes = 5
    return max(1, minutes) * 60


def _next_run_at(interval_seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, interval_seconds))).isoformat()


def _token_batches(tokens: list[str], batch_size: int = ACCOUNT_AUTO_REFRESH_BATCH_SIZE) -> list[list[str]]:
    size = max(1, int(batch_size or 1))
    return [tokens[index:index + size] for index in range(0, len(tokens), size)]


def start_limited_account_watcher(stop_event: Event) -> Thread:
    def worker() -> None:
        while not stop_event.is_set():
            processed = 0
            refreshed = 0
            errors: list[object] = []
            try:
                limited_tokens = account_service.list_limited_tokens()
                normal_tokens = account_service.list_normal_tokens()
                expiring_tokens = account_service.list_expiring_access_tokens()
                keepalive_tokens = account_service.list_refresh_token_keepalive_tokens()
                tokens = list(dict.fromkeys(account_service.list_tokens()))
                batches = _token_batches(tokens)
                account_service.start_auto_refresh_status(
                    total=len(tokens),
                    batch_size=ACCOUNT_AUTO_REFRESH_BATCH_SIZE,
                    batch_count=len(batches),
                )
                expiring_token_set = set(expiring_tokens)
                keepalive_tokens = [token for token in keepalive_tokens if token not in expiring_token_set]
                if tokens:
                    print(
                        "[account-watcher] refreshing "
                        f"{len(tokens)} accounts in batches of {ACCOUNT_AUTO_REFRESH_BATCH_SIZE} "
                        f"({len(normal_tokens)} normal, {len(limited_tokens)} limited, "
                        f"{len(expiring_tokens)} expiring access tokens)"
                    )
                    for batch_index, batch in enumerate(batches, start=1):
                        if stop_event.is_set():
                            break
                        result = account_service.refresh_accounts(batch)
                        processed += len(batch)
                        refreshed += int(result.get("refreshed") or 0)
                        batch_errors = result.get("errors") if isinstance(result, dict) else []
                        if isinstance(batch_errors, list):
                            errors.extend(batch_errors)
                        account_service.update_auto_refresh_status(
                            processed=processed,
                            refreshed=refreshed,
                            failed=len(errors),
                            batch_index=batch_index,
                        )
                else:
                    account_service.update_auto_refresh_status(processed=0, refreshed=0, failed=0, batch_index=0)
                if keepalive_tokens:
                    print(f"[account-watcher] keepalive {len(keepalive_tokens)} refresh tokens")
                    result = account_service.keepalive_refresh_tokens(keepalive_tokens)
                    if result.get("errors"):
                        print(f"[account-watcher] keepalive errors: {result['errors']}")
                interval_seconds = _account_refresh_interval_seconds()
                stopped = stop_event.is_set() and processed < len(tokens)
                account_service.finish_auto_refresh_status(
                    success=not errors and not stopped,
                    processed=processed,
                    refreshed=refreshed,
                    failed=len(errors),
                    error="stopped" if stopped else "",
                    next_run_at="" if stopped else _next_run_at(interval_seconds),
                )
            except Exception as exc:
                interval_seconds = _account_refresh_interval_seconds()
                account_service.finish_auto_refresh_status(
                    success=False,
                    processed=processed,
                    refreshed=refreshed,
                    failed=len(errors),
                    error=str(exc),
                    next_run_at=_next_run_at(interval_seconds),
                )
                print(f"[account-watcher] fail {exc}")
            interval_seconds = _account_refresh_interval_seconds()
            stop_event.wait(interval_seconds)

    thread = Thread(target=worker, name="account-watcher", daemon=True)
    thread.start()
    return thread


def resolve_web_asset(requested_path: str) -> Path | None:
    if not WEB_DIST_DIR.exists():
        return None
    clean_path = requested_path.strip("/")
    base_dir = WEB_DIST_DIR.resolve()
    candidates = [base_dir / "index.html"] if not clean_path else [
        base_dir / Path(clean_path),
        base_dir / clean_path / "index.html",
        base_dir / f"{clean_path}.html",
    ]
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(base_dir)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None
