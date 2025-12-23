"""清洗步骤：对抽取得到的文档文本进行规则化清洗与打分"""

from dataclasses import dataclass
from typing import Tuple

import ray.data as rd

from llm_data_pipeline.clean.rules import CleanRules, basic_clean, judge


@dataclass(frozen=True)
class CleanConfig:
    """清洗过程的配置项"""

    batch_size: int = 256
    rules: CleanRules = CleanRules()


def clean_dataset(
    ds: rd.Dataset, cfg: CleanConfig, taskpool_size: int = 0, num_cpus: float = 1.0
) -> Tuple[rd.Dataset, rd.Dataset]:
    """对输入 Dataset 批处理清洗并返回保留与丢弃的两个子集

    输入：ingest 输出的 Dataset（含 doc_id/url/warc_date/source_path/text）
    输出：`(kept_ds, dropped_ds)` 两个 Dataset
    """
    compute = rd.TaskPoolStrategy(size=taskpool_size) if taskpool_size else None

    def clean_batch(df):
        """单批次清洗逻辑，返回包含清洗结果与度量的 DataFrame"""

        texts = df["text"].astype(str).tolist()
        kept, reason = [], []
        m_non_ws, m_alpha_cjk, m_punct, m_dup = [], [], [], []

        cleaned = []
        for t in texts:
            t2 = basic_clean(t)
            ok, rsn, m = judge(t2, cfg.rules)
            cleaned.append(t2)
            kept.append(bool(ok))
            reason.append(rsn)
            m_non_ws.append(float(m.get("non_ws", 0.0)))
            m_alpha_cjk.append(float(m.get("alpha_cjk", 0.0)))
            m_punct.append(float(m.get("punct", 0.0)))
            m_dup.append(float(m.get("dup_line", 0.0)))

        out = df.copy()
        out["text"] = cleaned  # 直接覆盖，省空间
        out["kept"] = kept
        out["drop_reason"] = reason
        out["m_non_ws"] = m_non_ws
        out["m_alpha_cjk"] = m_alpha_cjk
        out["m_punct"] = m_punct
        out["m_dup_line"] = m_dup
        return out

    ds2 = ds.map_batches(
        clean_batch,
        batch_format="pandas",
        batch_size=cfg.batch_size,
        compute=compute,
        num_cpus=num_cpus,
    )

    kept_ds = ds2.filter(lambda r: r["kept"] is True)
    drop_ds = ds2.filter(lambda r: r["kept"] is False)
    return kept_ds, drop_ds
