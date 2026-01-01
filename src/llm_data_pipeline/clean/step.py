"""
Cleaning Step Logic.

This module implements the Ray Data specific logic for the cleaning step.
It defines the `CleanConfig` and the `clean_dataset` transformation function that applies the
cleaning rules in a distributed manner.
"""

from dataclasses import dataclass

import ray.data as rd

from llm_data_pipeline.clean.rules import CleanRules, basic_clean, judge


@dataclass(frozen=True)
class CleanConfig:
    """
    Configuration for the cleaning step.

    Attributes:
        batch_size: Batch size for Ray Data processing.
        rules: Set of rules for judging text quality.
    """

    batch_size: int = 256
    rules: CleanRules = CleanRules()


def clean_dataset(
    ds: rd.Dataset, cfg: CleanConfig, taskpool_size: int = 0, num_cpus: float = 1.0
) -> tuple[rd.Dataset, rd.Dataset]:
    """
    Cleans the input dataset based on the provided configuration rules.

    It separates the dataset into two: one with documents that passed the cleaning rules ('kept')
    and one with documents that failed ('dropped'), including the reason for rejection.

    Args:
        ds: Input Ray Dataset containing raw documents.
        cfg: Cleaning configuration.
        taskpool_size: Size of the actor pool for parallel processing (0=auto).
        num_cpus: Number of CPUs to allocate per task.

    Returns:
        A tuple (kept_ds, dropped_ds).
    """
    # map_batches default compute is "tasks"
    # we use concurrency to control parallelism if taskpool_size is set
    concurrency = taskpool_size if taskpool_size > 0 else None

    def clean_batch(df):
        """
        Processes a single batch of data.

        - Normalizes text.
        - Applies cleaning rules.
        - Calculates quality metrics.
        - Returns a DataFrame with added metadata columns (kept, drop_reason, metrics).
        """

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
        concurrency=concurrency,
        num_cpus=num_cpus,
    )

    kept_ds = ds2.filter(lambda r: r["kept"] is True)
    drop_ds = ds2.filter(lambda r: r["kept"] is False)
    return kept_ds, drop_ds
