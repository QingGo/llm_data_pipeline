from dataclasses import dataclass

# 修复fasttext的numpy 2.0兼容性问题
# 解决方案：直接替换fasttext模块中的predict方法实现
import fasttext
import fasttext.FastText
import fasttext.FastText as FastText_module
import numpy as np
from fasttext.FastText import _FastText

from llm_data_pipeline.core import PipelineLogger

# 保存原始的check函数
# Try to get the original check function from FastText module
original_check = None
for item in dir(FastText_module):
    if item == "check":
        original_check = getattr(FastText_module, item)
        break

# If original_check is still None after the loop, use this fallback
# We'll define a function and assign it directly to avoid redeclaration
if original_check is None:
    # Define the fallback function
    def _check_func(entry: str) -> str:
        """Default check function if original is not found"""
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\n')")
        return entry + "\n"

    # Assign the function to original_check
    original_check = _check_func


# 重新实现FastText._FastText.predict方法
def fixed_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    """Fixed predict method that returns simple lists instead of numpy arrays"""
    # Ensure original_check is a callable (should never be None due to our setup)
    assert callable(original_check), "original_check must be a callable"

    if isinstance(text, list):
        # 批量处理 - 使用FastText的multilinePredict方法
        texts = [original_check(t) for t in text]
        all_labels = []
        all_probs = []

        # 调用multilinePredict方法
        results = self.f.multilinePredict(texts, k, threshold, on_unicode_error)

        # 处理返回结果
        if isinstance(results, tuple) and len(results) == 2:
            # 格式：(all_labels, all_probs)
            labels_list, probs_list = results
            for labels, probs in zip(labels_list, probs_list, strict=True):
                all_labels.append(labels)
                all_probs.append([float(p) for p in probs])
        else:
            # 处理单个结果的情况
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    # 格式：(labels, probs)
                    labels, probs = result
                    all_labels.append(labels)
                    all_probs.append([float(p) for p in probs])
                elif isinstance(result, tuple) and len(result) == 1:
                    # 格式：((prob, label),)
                    prob_label_pairs = result[0]
                    labels = []
                    probs = []
                    for prob, label in prob_label_pairs:
                        labels.append(label)
                        probs.append(float(prob))
                    all_labels.append(labels)
                    all_probs.append(probs)

        return all_labels, all_probs
    else:
        # 单个文本处理
        text = original_check(text)

        # 调用predict方法
        result = self.f.predict(text, k, threshold, on_unicode_error)

        labels = []
        probs = []

        if isinstance(result, tuple):
            if len(result) == 2:
                # 格式：(labels, probs)
                labels, probs = result
                probs = [float(p) for p in probs]
            elif len(result) == 1:
                # 格式：((prob, label),)
                prob_label_pairs = result[0]
                for prob, label in prob_label_pairs:
                    labels.append(label)
                    probs.append(float(prob))
        elif isinstance(result, list):
            # 格式：[(prob, label)]
            for prob, label in result:
                labels.append(label)
                probs.append(float(prob))

        return labels, probs


# 应用补丁到_FastText类
_FastText.predict = fixed_predict


@dataclass
class QualityFilter:
    model_path: str
    threshold: float = 0.5
    pos_label: str | None = None  # 先跑一次打印 labels 再填

    def __post_init__(self) -> None:
        self.logger = PipelineLogger.get()
        self.logger.info(f"Loading fastText model: {self.model_path}")
        self.model = fasttext.load_model(self.model_path)

        self.all_labels: list[str] = list(map(str, self.model.get_labels()))
        if not self.all_labels:
            raise RuntimeError("No labels found in the model.")
        self.logger.info(f"Model labels: {self.model.get_labels()}")
        # 如果未指定正例标签，先给出提示，避免默默猜错
        if self.pos_label is None:
            raise ValueError("pos_label must be set to a valid label.")

    @staticmethod
    def normalize(text: str) -> str:
        # fastText 对换行较敏感，通常建议先抹平
        return " ".join(text.split())

    def score(self, text: str) -> float:
        text = self.normalize(text)
        if not text:
            return 0.0

        # 取全量 label 的分数，避免 k 设置不够导致取不到目标 label
        labels, probs = self.model.predict(text, k=len(self.all_labels) + 1)
        score_map: dict[str, float] = dict(zip(labels, probs, strict=True))
        pos_label = self.pos_label
        if pos_label is None:
            raise ValueError("pos_label must be set to a valid label.")

        return float(score_map.get(pos_label, 0.0))

    def keep(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    def filter_batch(self, texts: list[str]) -> list[str]:
        kept: list[str] = []
        for t in texts:
            if self.keep(t):
                kept.append(t)
        return kept


if __name__ == "__main__":
    samples = [
        # --- 正例：高质量百科/学术文本 (预期保留) ---
        "ABOUT AWB KIDS ARE KIDS JOIN THE CAST << Back to AWB News "
        "Christine Rouse is honored on the “Today Show” The executive director "
        "of Acting Without Boundaries (AWB), Christine Rouse, was featured on "
        "the NBC Today Show with “Kathie Lee and Hoda” on March 1, 2012. The "
        "monthly segment, called “Everyone Has A Story,” features one ordinary "
        "person that has had a life-changing experience in their own life. "
        "Christine submitted an essay describing her life’s mission of increasing "
        "awareness of and support for people with disabilities. She described "
        "the process of creating the two non-profits she manages – “Kids are "
        "Kids,” which provides disability awareness workshops and AWB which "
        "provides theater arts opportunities for children, youth and young "
        "adults with physical disabilities. Christine talked about the importance "
        "of both in increasing inclusion for people, especially young people, "
        "with physical disabilities. The March “Everyone Has A Story” segment "
        "featured Christine, her mother, and her brother. Christine’s mother "
        "read a letter she wrote about Christine’s life and the pride she takes "
        "in her many accomplishments. John Tartaglia, a Broadway performer sang "
        "a song written for Christine – “Different is Beautiful”- by Kathie Lee "
        "Gifford and David Freidman. The song has a powerful message and will "
        "be performed by AWB actors in the near future. To cap off this exciting "
        "experience, the Today Show honored Christine’s work with $1000 donations "
        "to each of her organizations, Kids are Kids and AWB. 750 E. Haverford "
        "Road, Bryn Mawr, PA 19010 Email: mmurphy@awb2004.org",
        "<li><b>astro-ph.EP - Earth and Planetary Astrophysics</b> "
        "(<a href='/list/astro-ph.EP/new'>new</a>, "
        "<a href='/list/astro-ph.EP/recent'>recent</a>, "
        "<a href='/list/astro-ph.EP/current'>current month</a>) "
        "<div class='description'>Interplanetary medium, planetary physics, "
        "planetary astrobiology, extrasolar planets, comets, asteroids, "
        "meteorites. Structure and formation of the solar system</div> </li>",
        "https://arxiv.org/abs/1706.03762 Abstract We propose the Transformer, "
        "a neural network architecture based solely on attention mechanisms, "
        "dispensing with recurrence and convolutions. We evaluate on machine "
        "translation benchmarks and analyze training cost, parallelization, and "
        "scaling behavior. References include BLEU, WMT, and attention ablations. "
        "© arXiv.org",
        "We study recurrent neural networks for sequence modeling and compare "
        "gated variants under a controlled experimental protocol. Across three "
        "benchmarks, the gated architectures converge faster and exhibit reduced "
        "gradient instability when the sequence length increases. Our "
        "implementation details and hyperparameters are reported to support "
        "reproducibility.",
        "This document specifies the behavior of a client and server during a "
        "handshake protocol that negotiates cryptographic parameters. "
        "Implementations MUST validate peer identities, enforce minimum key "
        "sizes, and reject deprecated ciphersuites. Security considerations "
        "discuss downgrade resilience and replay prevention.",
        "The survey estimates the monthly unemployment rate using a stratified "
        "sample and applies seasonal adjustment to improve comparability across "
        "years. Sampling error and nonresponse bias are quantified, and confidence "
        "intervals are provided for key aggregates. Methodological changes are "
        "documented to ensure continuity.",
        "Let (x_t) denote the hidden state and (y_t) the observation at time "
        "(t). Under linear dynamics with Gaussian noise, the Kalman filter yields "
        "the minimum mean-square error estimate and a closed-form update for the "
        "posterior covariance. The derivation follows directly from conditional "
        "Gaussian identities.",
        "Authentication assurance requires resistance to phishing, replay, and "
        "credential stuffing attacks. Verifiers should rate-limit failed attempts, "
        "bind sessions to cryptographic tokens, and log high-risk events for "
        "audit. Recovery mechanisms must avoid weakening the primary "
        "authenticator.",
        "The instrument measures spectral radiance over 400–700 nm with a spectral "
        "resolution of 1 nm and a radiometric uncertainty below 2% (k=2). "
        "Calibration is performed against a traceable reference source, and drift "
        "is corrected using daily dark frames. Raw data and processing scripts "
        "are archived.",
        "For adults with community-acquired pneumonia, empiric therapy should be "
        "selected based on local resistance patterns and patient risk factors. "
        "Treatment duration is typically guided by clinical stability and symptom "
        "resolution rather than a fixed number of days. Adverse events and "
        "contraindications are summarized for common regimens.",
        "The device operates from 1.8 V to 3.3 V and supports a maximum clock "
        "frequency of 80 MHz under typical conditions. Power consumption scales "
        "approximately linearly with frequency, and thermal derating applies "
        "above 85°C ambient. Electrical characteristics are specified across "
        "process and temperature corners.",
        "We preprocess text by normalizing Unicode, removing boilerplate "
        "navigation elements, and segmenting into paragraphs before feature "
        "extraction. The classifier is trained with subword n-grams to improve "
        "robustness to rare tokens and spelling variants. Evaluation includes "
        "both in-domain and out-of-domain validation sets.",
        "This dataset contains annotated time-series recordings collected under "
        "an approved protocol, with consent and anonymization procedures "
        "described. Labels were assigned by two independent raters and "
        "adjudicated by a third in cases of disagreement. Versioning and "
        "checksum files are provided for integrity verification.",
        # --- 反例：低质量广告/垃圾文本 (预期丢弃) ---
        "Buy cheap watches now!!! Click here!!! best price $9.99 for rolex replica.",
        "sex dating hot singles in your area! Sign up for free today. 100% free no credit card needed.",
        "viagra cialis levitra pharmacy online drug store... cheap prices fast delivery",
    ]

    qf = QualityFilter(
        model_path="./models/wikiref/en_model.bin",
        threshold=0.4,
        pos_label="__label__wiki",  # 设置为 '__label__wiki' 以保留高质量内容
    )

    for text in samples:
        s = qf.score(text)
        status = "✅ 保留" if s >= qf.threshold else "❌ 丢弃"
        qf.logger.info(f"{status} | score={s:.4f} | {text[:80]}...")


@dataclass
class LanguageFilter:
    model_path: str
    allowed_langs: list[str]  # e.g. ["zh", "en"] (automatic prefix handling)
    threshold: float = 0.4

    def __post_init__(self) -> None:
        # Use actor logger for Ray compatibility
        from llm_data_pipeline.core import get_actor_logger

        self.logger = get_actor_logger("llm_data_pipeline.quality")

        self.logger.info(f"Loading fastText LID model: {self.model_path}")
        self.model = fasttext.load_model(self.model_path)
        # Normalize allowed langs to fasttext label format
        self.allowed_labels = set()
        for lang in self.allowed_langs:
            if not lang.startswith("__label__"):
                lang = f"__label__{lang}"
            self.allowed_labels.add(lang)
        self.logger.info(f"Allowed labels: {self.allowed_labels}")

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.split())

    def predict(self, text: str) -> tuple[str, float]:
        text = self.normalize(text)
        if not text:
            return ("__label__unknown", 0.0)

        try:
            # 直接使用规范化后的文本，不要添加换行符
            # 经过patch的predict方法会处理换行符要求
            labels, probs = self.model.predict(text, k=1)

            # 确保返回值是有效的
            if isinstance(labels, list) and len(labels) > 0:
                label = labels[0]

                # 处理probs，兼容不同格式
                if isinstance(probs, list):
                    # 列表格式
                    if probs and len(probs) > 0:
                        # 检查probs[0]的类型
                        if isinstance(probs[0], (list, np.ndarray)):
                            prob = float(probs[0][0])
                        else:
                            prob = float(probs[0])
                    else:
                        prob = 0.0
                elif isinstance(probs, np.ndarray):
                    # numpy数组格式
                    prob = float(probs[0]) if probs.size > 0 else 0.0
                else:
                    # 其他格式
                    prob = 0.0

                self.logger.debug(f"Predicted language: {label} with probability: {prob:.4f}")
                return label, prob
        except Exception as e:
            self.logger.error(f"Error in language prediction: {e}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")

        return ("__label__unknown", 0.0)

    def keep(self, text: str) -> tuple[bool, str, float]:
        """Returns (keep, lang_label, score)"""
        label, score = self.predict(text)
        if label in self.allowed_labels and score >= self.threshold:
            return True, label, score
        return False, label, score
