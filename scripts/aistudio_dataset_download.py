#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百度 AI Studio 数据集高速下载（断点续传 / 多文件并行 / 签名 URL 过期自动刷新）

设计要点：
- 列表：POST https://aistudio.baidu.com/studio/dataset/detail
- 取链：GET  https://aistudio.baidu.com/llm/files/datasets/{id}/file/{fileId}/download
- 令牌：AISTUDIO_ACCESS_TOKEN 或 ~/.cache/aistudio/.auth/token 或 ~/.aistudio_token
- 分片：HTTP Range（BOS 实测支持 Accept-Ranges: bytes）
- 大文件：多连接分片并行（默认可调），每分片独立断点
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_HOST = os.environ.get("AISTUDIO_API_HOST", "https://aistudio.baidu.com")
CHUNK = 4 * 1024 * 1024
DEFAULT_PART = 32 * 1024 * 1024  # 每分片 32MB 并行


def _load_token() -> str:
    t = os.environ.get("AISTUDIO_ACCESS_TOKEN", "").strip()
    if t:
        return t
    for p in (
        Path.home() / ".cache" / "aistudio" / ".auth" / "token",
        Path.home() / ".aistudio_token",
    ):
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "未找到访问令牌：请设置环境变量 AISTUDIO_ACCESS_TOKEN，"
        "或写入 ~/.cache/aistudio/.auth/token"
    )


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "POST"),
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def fetch_file_ids(sess: requests.Session, token: str, dataset_id: int) -> List[Dict[str, Any]]:
    url = f"{API_HOST}/studio/dataset/detail"
    r = sess.post(
        url,
        data={"datasetId": dataset_id},
        headers={"Authorization": token},
        timeout=120,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("errorCode") != 0:
        raise RuntimeError(js.get("errorMsg", js))
    result = js.get("result") or {}
    ids = result.get("fileIds") or []
    meta = {m["fileId"]: m for m in (result.get("fileList") or []) if isinstance(m, dict)}
    out = []
    for fid in ids:
        m = meta.get(fid, {})
        out.append(
            {
                "fileId": int(fid),
                "fileName": m.get("fileName", f"file_{fid}"),
            }
        )
    return out


def fetch_download_url(
    sess: requests.Session, token: str, dataset_id: int, file_id: int
) -> str:
    url = f"{API_HOST}/llm/files/datasets/{dataset_id}/file/{file_id}/download"
    r = sess.get(url, headers={"Authorization": token}, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("errorCode") != 0:
        raise RuntimeError(js.get("errorMsg", js))
    return js["result"]["fileUrl"]


def _head_size(sess: requests.Session, file_url: str) -> int:
    h = sess.head(file_url, allow_redirects=True, timeout=120)
    h.raise_for_status()
    cl = h.headers.get("Content-Length")
    if not cl:
        return 0
    return int(cl)


def _download_part(
    sess: requests.Session,
    file_url: str,
    out_path: Path,
    start: int,
    end: int,
    pbar_lock: threading.Lock,
    pbar,
) -> None:
    part_path = Path(str(out_path) + f".part_{start}_{end}")
    cur = part_path.stat().st_size if part_path.exists() else 0
    rel_start = start + cur
    if rel_start > end:
        return
    headers = {"Range": f"bytes={rel_start}-{end}"}
    mode = "ab" if cur else "wb"
    with sess.get(
        file_url, headers=headers, stream=True, timeout=(30, 300)
    ) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        with open(part_path, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                if pbar and pbar_lock:
                    with pbar_lock:
                        pbar.update(len(chunk))


def _merge_parts(out_path: Path, ranges: List[Tuple[int, int]]) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".merging")
    with open(tmp, "wb") as out:
        for start, end in ranges:
            part_path = Path(str(out_path) + f".part_{start}_{end}")
            with open(part_path, "rb") as p:
                while True:
                    b = p.read(CHUNK)
                    if not b:
                        break
                    out.write(b)
            part_path.unlink(missing_ok=True)
    tmp.replace(out_path)


def download_file_parallel(
    sess: requests.Session,
    file_url: str,
    dest: Path,
    total: int,
    workers: int,
    part_size: int,
    log: logging.Logger,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == total:
        log.info("已存在且大小正确，跳过: %s", dest.name)
        return

    ranges: List[Tuple[int, int]] = []
    s = 0
    while s < total:
        e = min(s + part_size - 1, total - 1)
        ranges.append((s, e))
        s = e + 1

    if len(ranges) <= 1 or workers <= 1:
        _download_single_stream(sess, file_url, dest, total, log)
        return

    try:
        from tqdm import tqdm
    except ImportError:
        pbar = None
        pbar_lock = threading.Lock()
    else:
        pbar_lock = threading.Lock()
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=True)

    with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as ex:
        futs = [
            ex.submit(_download_part, sess, file_url, dest, a, b, pbar_lock, pbar)
            for a, b in ranges
        ]
        for fu in as_completed(futs):
            fu.result()
    if pbar:
        pbar.close()
    _merge_parts(dest, ranges)
    if dest.stat().st_size != total:
        raise IOError(f"合并后大小不符: {dest}")


def _download_single_stream(
    sess: requests.Session,
    file_url: str,
    dest: Path,
    total: int,
    log: logging.Logger,
) -> None:
    """单连接 Range 续传（与官方 SDK 思路一致）"""
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    cur = part.stat().st_size if part.exists() else 0
    if total and cur >= total:
        part.replace(dest)
        return
    headers = {}
    mode = "ab"
    if cur == 0:
        mode = "wb"
    else:
        headers["Range"] = f"bytes={cur}-"
    with sess.get(
        file_url, headers=headers, stream=True, timeout=(30, 600)
    ) as r:
        if r.status_code == 416:
            part.unlink(missing_ok=True)
            cur = 0
            mode = "wb"
            r = sess.get(file_url, stream=True, timeout=(30, 600))
        r.raise_for_status()
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
    if total and part.stat().st_size != total:
        log.warning("大小预期 %s 实际 %s: %s", total, part.stat().st_size, part)
    part.replace(dest)


def download_one_file(
    sess: requests.Session,
    token: str,
    dataset_id: int,
    file_id: int,
    name: str,
    out_dir: Path,
    state: Dict[str, Any],
    parallel_parts: int,
    part_mb: int,
    log: logging.Logger,
) -> None:
    key = str(file_id)
    st = state.setdefault("files", {})
    done = st.get(key, {}).get("done")
    dest = out_dir / name
    if done and dest.is_file() and dest.stat().st_size == done.get("size", -1):
        log.info("状态已完成，跳过: %s", name)
        return

    for attempt in range(8):
        try:
            file_url = fetch_download_url(sess, token, dataset_id, file_id)
            total = _head_size(sess, file_url)
            log.info("开始下载 %s (%d bytes)", name, total)

            if parallel_parts > 1 and total > part_mb * 1024 * 1024:
                download_file_parallel(
                    sess,
                    file_url,
                    dest,
                    total,
                    workers=parallel_parts,
                    part_size=part_mb * 1024 * 1024,
                    log=log,
                )
            else:
                _download_single_stream(sess, file_url, dest, total, log)

            sz = dest.stat().st_size
            st[key] = {"done": True, "size": sz, "name": name}
            _save_state(out_dir, state)
            log.info("完成: %s (%d bytes)", name, sz)
            return
        except Exception as e:
            log.exception("第 %d 次失败 %s: %s", attempt + 1, name, e)
            time.sleep(min(60, 2**attempt))

    raise RuntimeError(f"下载失败: {name}")


def _state_path(out_dir: Path) -> Path:
    return out_dir / ".aistudio_download_state.json"


def _load_state(out_dir: Path) -> Dict[str, Any]:
    p = _state_path(out_dir)
    if not p.is_file():
        return {"version": 1, "files": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_state(out_dir: Path, state: Dict[str, Any]) -> None:
    p = _state_path(out_dir)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="百度 AI Studio 数据集下载（断点续传）")
    ap.add_argument("--dataset-id", type=int, default=35660)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/root/autodl-tmp/data/nih-chest-xrays-data"),
    )
    ap.add_argument("--parallel-files", type=int, default=2, help="同时下载的文件数")
    ap.add_argument(
        "--parallel-parts",
        type=int,
        default=4,
        help="单文件分片并行数（大文件）",
    )
    ap.add_argument(
        "--part-mb",
        type=int,
        default=32,
        help="超过此大小才启用分片并行（MB）",
    )
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "aistudio_download.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("aistudio_dl")

    token = _load_token()
    sess = _session()
    files = fetch_file_ids(sess, token, args.dataset_id)
    log.info("数据集 %s 共 %d 个文件", args.dataset_id, len(files))

    state = _load_state(out_dir)

    def run_one(meta: Dict[str, Any]) -> None:
        download_one_file(
            sess,
            token,
            args.dataset_id,
            meta["fileId"],
            meta["fileName"],
            out_dir,
            state,
            parallel_parts=args.parallel_parts,
            part_mb=args.part_mb,
            log=log,
        )

    with ThreadPoolExecutor(max_workers=args.parallel_files) as ex:
        futs = [ex.submit(run_one, f) for f in files]
        for fu in as_completed(futs):
            fu.result()

    log.info("全部完成，输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
