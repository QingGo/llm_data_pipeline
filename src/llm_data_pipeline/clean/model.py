import fasttext
import fasttext.FastText
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# Monkey patch fasttext to fix numpy 2.0 compatibility
# fasttext 0.9.3 uses np.array(probs, copy=False) which fails in numpy 2.0
# when the array needs to be copied (e.g. from list).
# We replace the predict method with a fixed version using np.asarray.
def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    def check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        entry += "\n"
        return entry

    if type(text) == list:
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(
            text, k, threshold, on_unicode_error
        )
        return all_labels, all_probs
    else:
        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = ([], ())

        # Fix for numpy 2.0: use np.asarray instead of np.array(..., copy=False)
        return labels, np.asarray(probs)


fasttext.FastText._FastText.predict = _patched_predict



@dataclass
class QualityFilter:
    model_path: str
    threshold: float = 0.5
    pos_label: Optional[str] = None  # 先跑一次打印 labels 再填

    def __post_init__(self) -> None:
        print(f"Loading fastText model: {self.model_path}")
        self.model = fasttext.load_model(self.model_path)

        self.all_labels: List[str] = list(self.model.get_labels())
        if not self.all_labels:
            raise RuntimeError("No labels found in the model.")

        # 如果未指定正例标签，先给出提示，避免默默猜错
        if self.pos_label is None:
            print("Model labels:", self.all_labels)
            print(
                "Tip: set pos_label to the actual positive label, e.g. '__label__wikiref' or similar."
            )

    @staticmethod
    def normalize(text: str) -> str:
        # fastText 对换行较敏感，通常建议先抹平
        return " ".join(text.split())

    def score(self, text: str) -> float:
        text = self.normalize(text)
        if not text:
            return 0.0

        # 取全量 label 的分数，避免 k 设置不够导致取不到目标 label
        labels, probs = self.model.predict(text, k=len(self.all_labels))
        score_map = dict(zip(labels, probs))

        if self.pos_label is None:
            # 没指定就退化成取 top1 分数，至少可用于粗排演示
            return float(probs[0])

        return float(score_map.get(self.pos_label, 0.0))

    def keep(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    def filter_batch(self, texts: List[str]) -> List[str]:
        kept: List[str] = []
        for t in texts:
            if self.keep(t):
                kept.append(t)
        return kept


if __name__ == "__main__":
    samples = [
        "Einstein developed the theory of relativity. [1] It is one of the pillars of modern physics.",
        "Buy cheap watches now!!! Click here!!! best price $9.99",
    ]

    qf = QualityFilter(
        model_path="./models/wikiref/en_model.bin",
        threshold=0.4,
        pos_label=None,  # 先跑一次观察 labels，再填成真实正例标签
    )

    for text in samples:
        s = qf.score(text)
        status = "✅ 保留" if s >= qf.threshold else "❌ 丢弃"
        print(f"{status} | score={s:.4f} | {text[:80]}...")
