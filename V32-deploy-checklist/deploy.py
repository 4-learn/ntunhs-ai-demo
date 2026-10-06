"""V32｜上線清單的程式部分：設定檔、模型 hash、結構化紀錄、啟動自檢、版本切換與回滾。

    from deploy import load_config, sha256_12, self_check, JsonLog, Releases

    cfg = load_config("V32-deploy-checklist/config/fall.yaml")   # pydantic 驗證，錯了就拒絕
    results = self_check(cfg, base=ROOT)                          # [(項目, ok, 說明)]；有 False 就不准啟動
    log = JsonLog("/tmp/ntunhs/V32/app.jsonl")                    # 一行一個 JSON
    log.write("startup", release=cfg.release, ok=True)

V28／V29 的管線在啟動時應呼叫 start(cfg, ...)：自檢失敗就丟 StartupRefused，不進主迴圈。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TW = timezone(timedelta(hours=8))


# ---------------------------------------------------------------- 設定檔


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    path: str                                        # 相對於 base（demo/ 或 release 資料夾）
    sha256_12: str = Field(pattern=r"^[0-9a-f]{12}$")
    card: str | None = None                          # 定版卡（例如 P16 的 config/fall.yaml），相對於 base


class Threshold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = Field(ge=0, le=1)
    chosen_on: str = Field(min_length=1)             # 依據：在哪份資料上選的（與 P16 定版卡同一組 key）
    rule: str = Field(min_length=1)                  # 依據：什麼成本、什麼格點
    approved_by: str = Field(min_length=1)
    approved_on: date


class Camera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    prefix: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    where: str


class Notify(BaseModel):
    model_config = ConfigDict(extra="forbid")
    push_score: float = Field(ge=0, le=1)
    escalate_min: float = Field(gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    app: str
    release: str
    source: Literal["pose", "defect"]
    site: str
    model: ModelFile
    threshold: Threshold
    cooldown_s: float = Field(ge=0)
    cameras: list[Camera] = Field(min_length=1)
    notify: Notify | None = None
    api_url: str
    log_path: str

    @field_validator("cameras")
    @classmethod
    def _unique(cls, v):
        ids = [c.id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"攝影機 id 重複：{ids}")
        return v


def load_config(path) -> AppConfig:
    """讀 YAML 並驗證；格式錯誤會丟 pydantic.ValidationError（指出哪個欄位）。"""
    with open(path, encoding="utf-8") as f:
        return AppConfig.model_validate(yaml.safe_load(f))


# ---------------------------------------------------------------- 模型 hash


def sha256_12(path, chunk: int = 1 << 20) -> str:
    """模型檔 SHA-256 的前 12 個 16 進位字元。分段讀，不把整個檔案載入記憶體。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- 結構化紀錄


class JsonLog:
    """JSON Lines：一行一筆，固定欄位 ts、level、event，其餘自由。clock 可換成假時鐘（講義輸出可重現）。"""

    def __init__(self, path, clock: Callable[[], datetime] = lambda: datetime.now(TW), echo=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.echo = clock, echo

    def write(self, event: str, level: str = "info", **fields) -> dict:
        rec = {"ts": self.clock().isoformat(timespec="seconds"), "level": level, "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line)
        return rec

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------- 啟動自檢


class StartupRefused(RuntimeError):
    pass


def self_check(cfg: AppConfig, base) -> list[tuple[str, bool, str]]:
    """回傳 [(項目, 是否通過, 說明)]。只檢查、不修正。"""
    base = Path(base)
    out = []
    mp = base / cfg.model.path
    if not mp.exists():
        out.append(("模型檔存在", False, f"找不到 {cfg.model.path}"))
        out.append(("模型 hash 相符", False, "沒有檔案可比"))
    else:
        out.append(("模型檔存在", True, cfg.model.path))
        got = sha256_12(mp)
        out.append(("模型 hash 相符", got == cfg.model.sha256_12,
                    f"設定 {cfg.model.sha256_12}，實際 {got}"))
    if cfg.model.card:
        cp = base / cfg.model.card
        if not cp.exists():
            out.append(("與定版卡一致", False, f"找不到 {cfg.model.card}"))
        else:
            card = yaml.safe_load(cp.read_text(encoding="utf-8"))["model"]
            same = (card["name"], card["sha256_12"]) == (cfg.model.name, cfg.model.sha256_12)
            out.append(("與定版卡一致", same, f"定版卡 {card['name']}@{card['sha256_12']}"))
    th = cfg.threshold
    out.append(("門檻已核准", bool(th.approved_by.strip()) and "待" not in th.approved_by,
                f"{th.value:.2f}（{th.approved_by}，{th.approved_on}）"))
    mock = os.environ.get("NTUNHS_MOCK") == "1"
    key = bool(os.environ.get("EZCAM_API_KEY"))
    out.append(("攝影機金鑰", mock or key, "mock 模式，不需要" if mock else ("已由環境變數提供" if key else "缺 EZCAM_API_KEY")))
    try:
        Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        out.append(("紀錄檔可寫", os.access(Path(cfg.log_path).parent, os.W_OK), "log_path 所在資料夾"))
    except OSError as e:
        out.append(("紀錄檔可寫", False, str(e)))
    return out


def start(cfg: AppConfig, base, log: JsonLog) -> None:
    """自檢＋寫紀錄；任何一項失敗就丟 StartupRefused（管線不准進主迴圈）。"""
    res = self_check(cfg, base)
    failed = [(k, msg) for k, ok, msg in res if not ok]
    if failed:
        log.write("startup_refused", level="error", app=cfg.app, release=cfg.release,
                  failed=[k for k, _ in failed], detail=failed[0][1])
        raise StartupRefused("；".join(f"{k}：{m}" for k, m in failed))
    log.write("startup", app=cfg.app, release=cfg.release, model=cfg.model.name,
              model_sha256_12=cfg.model.sha256_12, threshold=cfg.threshold.value, cameras=len(cfg.cameras))


# ---------------------------------------------------------------- 版本與回滾


class Releases:
    """releases/{版本}/ 各放一份 config.yaml＋模型檔；CURRENT 檔寫目前版本，HISTORY 記每次切換。

    切換一律「先自檢新版本，通過才改 CURRENT」；回滾＝切回上一個通過自檢的版本。
    """

    def __init__(self, root, log: JsonLog):
        self.root, self.log = Path(root), log
        (self.root / "releases").mkdir(parents=True, exist_ok=True)

    def dir(self, version: str) -> Path:
        return self.root / "releases" / version

    def add(self, version: str, config: dict, model_src, card_src=None) -> Path:
        d = self.dir(version)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_src, d / Path(config["model"]["path"]).name)
        model = {**config["model"], "path": Path(config["model"]["path"]).name}
        if card_src is not None:                     # 定版卡跟著版本走
            shutil.copyfile(card_src, d / "card.yaml")
            model["card"] = "card.yaml"
        config = {**config, "release": version, "model": model}
        (d / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return d

    def current(self) -> str | None:
        f = self.root / "CURRENT"
        return f.read_text().strip() if f.exists() else None

    def history(self) -> list[str]:
        f = self.root / "HISTORY"
        return f.read_text().split() if f.exists() else []

    def activate(self, version: str, by: str, reason: str) -> bool:
        cfg = load_config(self.dir(version) / "config.yaml")
        try:
            start(cfg, self.dir(version), self.log)
        except StartupRefused as e:
            self.log.write("activate_failed", level="error", release=version, by=by, kept=self.current(), detail=str(e))
            return False
        prev = self.current()
        (self.root / "CURRENT").write_text(version + "\n")
        with open(self.root / "HISTORY", "a") as f:
            f.write(version + "\n")
        self.log.write("activate", release=version, previous=prev, by=by, reason=reason)
        return True

    def rollback(self, by: str, reason: str) -> str | None:
        """切回 HISTORY 裡「目前版本之前」最近的一個版本。"""
        hist, cur = self.history(), self.current()
        for v in reversed(hist[:-1]):
            if v != cur:
                return v if self.activate(v, by=by, reason=f"rollback：{reason}") else None
        return None


__all__ = ["AppConfig", "load_config", "sha256_12", "JsonLog", "self_check", "start", "StartupRefused",
           "Releases", "ValidationError", "TW"]
