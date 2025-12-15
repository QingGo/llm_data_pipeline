"""WARC/WET 文件抽取步骤

将 *.wet.gz 解析为标准化的文本文档条目，进行轻量清洗与长度裁剪。
"""

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from warcio.archiveiterator import ArchiveIterator


@dataclass(frozen=True)
class IngestConfig:
    """抽取管道的配置项"""
    min_text_chars: int = 200
    max_text_chars: int = 200_000
    max_docs_per_file: int = 200  # 0=不限（debug 时可设 200）
    warc_type_keep: str = "conversion"  # WET 常用 conversion


def _normalize_text(text: str) -> str:
    """对文本做轻量级规范化，统一换行并压缩多余空行"""
    # 轻量 normalize
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # 压缩超多空行
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _doc_id(source_path: str, url: str, warc_date: str, record_id: str) -> str:
    """根据来源路径、URL、WARC 日期与记录 ID 生成稳定的文档哈希 ID"""
    raw = f"{source_path}\n{url}\n{warc_date}\n{record_id}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def extract_wet_gz_file(path: Path, cfg: IngestConfig) -> List[Dict[str, Any]]:
    """
    将一个 *.wet.gz 文件解析为多条标准化文档记录。

    参数:
      path: 输入的 `*.wet.gz` 文件路径
      cfg: `IngestConfig` 配置对象

    返回:
      多条文档记录的列表。每条记录包含：
        - doc_id: 稳定的 SHA1 文档 ID
        - url: 来源 URL
        - warc_date: WARC 记录日期
        - source_path: 原始文件路径
        - text: 经过轻量规范化与长度裁剪的正文
    """
    docs: List[Dict[str, Any]] = []
    n = 0

    with path.open("rb") as f:
        with gzip.GzipFile(fileobj=f) as gz:
            for rec in ArchiveIterator(gz):
                rtype = (rec.rec_headers.get_header("WARC-Type") or "").lower()
                if rtype != cfg.warc_type_keep:
                    continue

                url = rec.rec_headers.get_header("WARC-Target-URI") or ""
                warc_date = rec.rec_headers.get_header("WARC-Date") or ""
                record_id = rec.rec_headers.get_header("WARC-Record-ID") or ""

                stream = rec.content_stream() or rec.raw_stream
                data = stream.read() if stream is not None else b""
                if not data:
                    continue

                text = data.decode("utf-8", errors="replace")
                text = _normalize_text(text)

                if len(text) < cfg.min_text_chars:
                    continue
                if len(text) > cfg.max_text_chars:
                    text = text[: cfg.max_text_chars]

                docs.append(
                    {
                        "doc_id": _doc_id(str(path), url, warc_date, record_id),
                        "url": url,
                        "warc_date": warc_date,
                        "source_path": str(path),
                        "text": text,
                    }
                )

                n += 1
                if cfg.max_docs_per_file and n >= cfg.max_docs_per_file:
                    break

    return docs
