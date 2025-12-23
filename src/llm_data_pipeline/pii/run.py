import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import ray
from ray.data import ActorPoolStrategy

# -------------------------
# RE2-compatible regexes (PyArrow uses RE2)
# NOTE: RE2 does NOT support lookaround. Keep patterns simple.
# e.g. alice.smith+news@sub.example.co.uk, a_b-1%2@test-domain.io
# -------------------------
EMAIL_RE = r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b"

# IPv4 (may overmatch 999.999.999.999; for aggressive redaction often acceptable)
# e.g. 192.168.1.1, 8.8.8.8
IPV4_RE = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

# IPv6 heuristic (RE2-friendly); not perfect but good enough for redaction
# e.g. 2001:0db8:85a3:0000:0000:8a2e:0370:7334, 2001:db8::1
IPV6_RE = r"\b(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}\b"

# Phone heuristic: keeps FP moderate, but still errs on recall (user prefers FP over FN)
# e.g. 212-555-1234, +1 212 555 1234, 12345678901
PHONE_RE = r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"

# US SSN (hyphenated only)
# e.g. 123-45-6789
SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"

# Cheap gating triggers (multi-lingual-ish) 粗筛
# English + Chinese contact keywords (extend as needed)
# e.g. contact me at ..., 电话：..., 微信：abc123
CONTACT_KW_RE = r"(?i)\b(contact|call|email\s+me|reach\s+me|tel|phone|ssn|wechat|whatsapp|line|telegram)\b|联系我|电话|手机号|邮箱|微信|QQ|WhatsApp|Telegram|Line"
# English "Full Name-like" shape (only used for gating, not final replacement)
# e.g. John Smith, Alice Johnson
NAME_SHAPE_EN_RE = r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"
# Chinese character heuristic (for cheap lang guess)
# e.g. 张三
HAS_CJK_RE = r"[\u4e00-\u9fff]"


@dataclass
class Config:
    text_col: str = "text"
    lang_col: Optional[str] = None  # if exists, use it to route NER language
    keep_stats: bool = False  # keep pii_* columns in output
    enable_person_ner: bool = True
    # NER languages to attempt (others skip PERSON redaction)
    supported_langs: Optional[List[str]] = None
    # Presidio spaCy model map
    spacy_models: Optional[Dict[str, str]] = None
    # Analyzer score threshold (lower => more recall)
    ner_score_threshold: float = 0.0


class StructuredPIIRedactor:
    """
    Fast path: Arrow vectorized redaction for EMAIL/IP/PHONE/SSN + gating signals.
    Produces:
      - redacted text
      - need_ner (bool)
      - ner_lang (string) if NER enabled (route language for Presidio)
      - optional stats columns if keep_stats enabled
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, batch: pa.Table) -> pa.Table:
        col = self.cfg.text_col
        if col not in batch.column_names:
            raise ValueError(f"Column '{col}' not found. Available: {batch.column_names}")

        text = batch[col]

        # --- Match signals (vectorized) ---
        has_email = pc.match_substring_regex(text, EMAIL_RE)
        has_ip4 = pc.match_substring_regex(text, IPV4_RE)
        has_ip6 = pc.match_substring_regex(text, IPV6_RE)
        has_phone = pc.match_substring_regex(text, PHONE_RE)
        has_ssn = pc.match_substring_regex(text, SSN_RE)

        # Cheap heuristics for gating (high recall)
        has_at = pc.match_substring(text, "@")
        has_kw = pc.match_substring_regex(text, CONTACT_KW_RE)
        has_name_shape = pc.match_substring_regex(text, NAME_SHAPE_EN_RE)

        # need_ner if any structured PII hit OR cheap "contact/name-ish" triggers
        any_structured = pc.or_(pc.or_(has_email, has_phone), pc.or_(pc.or_(has_ssn, has_ip4), has_ip6))
        need_ner = pc.or_(any_structured, pc.or_(has_at, pc.or_(has_kw, has_name_shape)))

        # --- Apply vectorized replacements (Arrow/RE2) ---
        red = pc.replace_substring_regex(text, EMAIL_RE, "<EMAIL>")
        red = pc.replace_substring_regex(red, IPV4_RE, "<IP>")
        red = pc.replace_substring_regex(red, IPV6_RE, "<IP>")
        red = pc.replace_substring_regex(red, PHONE_RE, "<PHONE>")
        red = pc.replace_substring_regex(red, SSN_RE, "<SSN>")

        # Replace column
        out = batch.set_column(batch.schema.get_field_index(col), col, red)

        # Add gating column
        out = out.append_column("need_ner", need_ner)

        # Decide NER language per row (if NER enabled)
        if self.cfg.enable_person_ner:
            ner_lang = self._infer_or_use_lang(out, red)
            out = out.append_column("ner_lang", ner_lang)

        # Optional stats (helpful for QA, but extra columns)
        if self.cfg.keep_stats:
            out = out.append_column("pii_has_email", has_email)
            out = out.append_column("pii_has_ip4", has_ip4)
            out = out.append_column("pii_has_ip6", has_ip6)
            out = out.append_column("pii_has_phone", has_phone)
            out = out.append_column("pii_has_ssn", has_ssn)

        return out

    def _infer_or_use_lang(self, batch: pa.Table, text_arr: pa.ChunkedArray | pa.Array) -> pa.Array:
        # If lang column exists, use it; else cheap heuristic: has CJK => zh, else en
        if self.cfg.lang_col and self.cfg.lang_col in batch.column_names:
            lang = batch[self.cfg.lang_col]
            # normalize a bit: take first two letters (en, zh, fr, ...)
            # If null/empty, fallback to heuristic.
            lang_py = lang.to_pylist()
            text_py = text_arr.to_pylist()
            out = []
            for l, t in zip(lang_py, text_py):
                if isinstance(l, str) and len(l) >= 2:
                    out.append(l[:2].lower())
                else:
                    out.append("zh" if _has_cjk(t) else "en")
            return pa.array(out, type=pa.string())
        else:
            # heuristic
            text_py = text_arr.to_pylist()
            out = ["zh" if _has_cjk(t) else "en" for t in text_py]
            return pa.array(out, type=pa.string())


def _has_cjk(s: Optional[str]) -> bool:
    if not s:
        return False
    # RE2 in Arrow is fine, but here we keep it python-level for per-row fallback only.
    # This function is used only if lang_col missing or null.
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            return True
    return False


class PresidioPersonNER:
    """
    Slow path: Presidio AnalyzerEngine + AnonymizerEngine for PERSON -> <NAME>.
    Use ActorPoolStrategy so models are loaded once per actor.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
        except Exception as e:
            raise RuntimeError(
                "Presidio not installed. Install:\n"
                "  pip install presidio-analyzer presidio-anonymizer spacy\n"
                "and download spaCy models you use (e.g. en_core_web_sm, zh_core_web_sm).\n"
            ) from e

        if not self.cfg.supported_langs:
            self.cfg.supported_langs = ["en", "zh"]
        if not self.cfg.spacy_models:
            self.cfg.spacy_models = {
                "en": "en_core_web_sm",
                "zh": "zh_core_web_sm",
            }

        # Build NLP engine with multi-language spaCy models
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": lang, "model_name": self.cfg.spacy_models[lang]}
                for lang in self.cfg.supported_langs
                if lang in self.cfg.spacy_models
            ],
        }
        provider = NlpEngineProvider(nlp_configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=self.cfg.supported_langs,
        )
        self.anonymizer = AnonymizerEngine()

        self.operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
        }
        self.entities = ["PERSON"]

    def __call__(self, batch: pa.Table) -> pa.Table:
        col = self.cfg.text_col
        if col not in batch.column_names:
            raise ValueError(f"Column '{col}' not found in batch.")
        if "ner_lang" not in batch.column_names:
            raise ValueError("Expected 'ner_lang' column for NER routing.")
        if "need_ner" not in batch.column_names:
            raise ValueError("Expected 'need_ner' column for gating.")

        text = batch[col].to_pylist()
        langs = batch["ner_lang"].to_pylist()
        need = batch["need_ner"].to_pylist()

        out_text = list(text)

        # Process row-by-row (Presidio analyze is per-text).
        # We skip rows not in need_ner or unsupported languages.
        for i, (t, lang, do_ner) in enumerate(zip(text, langs, need)):
            if not do_ner or not isinstance(t, str) or not t:
                continue
            if not isinstance(lang, str) or lang not in self.cfg.supported_langs:
                continue

            try:
                results = self.analyzer.analyze(
                    text=t,
                    entities=self.entities,
                    language=lang,
                    score_threshold=self.cfg.ner_score_threshold,
                )
                if results:
                    out_text[i] = self.anonymizer.anonymize(
                        text=t,
                        analyzer_results=results,
                        operators=self.operators,
                    ).text
            except Exception:
                # For production you might want to log these to a side channel.
                # Here we fail-soft to keep pipeline going.
                continue

        new_col = pa.array(out_text, type=pa.string())
        return batch.set_column(batch.schema.get_field_index(col), col, new_col)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ray Data + Arrow/RE2 + Presidio PII redaction to Parquet.")
    p.add_argument(
        "--input", default="outputs/dev/cleaned_parquet", help="Input parquet path (local://, s3://, gs://, etc.)"
    )
    p.add_argument("--output", default="outputs/dev/pii_parquet", help="Output parquet directory")
    p.add_argument("--text-col", default="text", help="Text column name (default: text)")
    p.add_argument(
        "--lang-col", default="", help="Optional language column (e.g. lang). If missing/empty, heuristic en/zh."
    )
    p.add_argument(
        "--num-blocks", type=int, default=0, help="Repartition to this many blocks (controls output file count)"
    )
    p.add_argument("--batch-size-structured", type=int, default=4096, help="Rows per batch for structured redaction")
    p.add_argument("--batch-size-ner", type=int, default=256, help="Rows per batch for NER stage (smaller is safer)")
    p.add_argument("--actors-ner", type=int, default=32, help="Actor pool size for Presidio NER stage")
    p.add_argument("--disable-ner", action="store_true", help="Disable PERSON NER stage entirely")
    p.add_argument("--keep-stats", action="store_true", help="Keep pii_has_* columns in output")
    p.add_argument("--supported-langs", default="en,zh", help="NER supported langs, comma-separated (default: en,zh)")
    p.add_argument("--spacy-en", default="en_core_web_sm", help="spaCy model for English")
    p.add_argument("--spacy-zh", default="zh_core_web_sm", help="spaCy model for Chinese")
    p.add_argument(
        "--ner-score-threshold", type=float, default=0.0, help="Presidio analyze score threshold (lower => more recall)"
    )
    p.add_argument("--ray-address", default=None, help="Ray cluster address (auto/local)")
    return p.parse_args()


def main():
    args = parse_args()

    ray.init(address=args.ray_address)

    cfg = Config(
        text_col=args.text_col,
        lang_col=(args.lang_col.strip() or None),
        keep_stats=args.keep_stats,
        enable_person_ner=(not args.disable_ner),
        supported_langs=[x.strip() for x in args.supported_langs.split(",") if x.strip()],
        spacy_models={
            "en": args.spacy_en,
            "zh": args.spacy_zh,
        },
        ner_score_threshold=args.ner_score_threshold,
    )

    ds = ray.data.read_parquet(args.input)

    if args.num_blocks and args.num_blocks > 0:
        ds = ds.repartition(args.num_blocks)

    # 1) Fast structured PII redaction + gating + ner_lang
    ds = ds.map_batches(
        StructuredPIIRedactor,
        fn_constructor_kwargs={"cfg": cfg},
        batch_format="pyarrow",
        batch_size=args.batch_size_structured,
        compute="tasks",
        zero_copy_batch=True,
    )

    if cfg.enable_person_ner:
        # 2) Split dataset: run NER only where need_ner==True AND lang supported
        supported = set(cfg.supported_langs)

        ds_ner = ds.filter(lambda r: bool(r["need_ner"]) and (r.get("ner_lang") in supported))
        ds_no = ds.filter(lambda r: (not bool(r["need_ner"])) or (r.get("ner_lang") not in supported))

        # 3) Presidio PERSON NER stage (actor pool to reuse models)
        ds_ner = ds_ner.map_batches(
            PresidioPersonNER,
            fn_constructor_kwargs={"cfg": cfg},
            batch_format="pyarrow",
            batch_size=args.batch_size_ner,
            compute=ActorPoolStrategy(size=args.actors_ner),
            zero_copy_batch=False,  # NER stage touches Python objects anyway
        )

        # 4) Merge back
        ds_out = ds_no.union(ds_ner)
    else:
        ds_out = ds

    # 5) Drop internal columns unless asked to keep
    drop_cols = ["need_ner", "ner_lang"]
    if not cfg.keep_stats:
        drop_cols += ["pii_has_email", "pii_has_ip4", "pii_has_ip6", "pii_has_phone", "pii_has_ssn"]

    # Only drop columns that exist
    existing = [c for c in drop_cols if c in ds_out.schema().names]
    if existing:
        ds_out = ds_out.drop_columns(existing)

    # 6) Write parquet
    ds_out.write_parquet(args.output)


if __name__ == "__main__":
    main()
