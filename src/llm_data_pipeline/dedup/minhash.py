import hashlib
import random
import re
from typing import Iterable, List, Sequence, Set, Tuple

from datasketch import MinHash

# --------- 1) normalize + shingling ---------


def normalize_text(text: str) -> str:
    # 你可以按需求加更多规则：Unicode NFKC、去HTML等
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngrams(text: str, n: int = 5) -> Set[bytes]:
    """
    中文/混合文本常用 char n-gram。
    返回 bytes，便于后续哈希与 datasketch.update 对齐。
    """
    t = normalize_text(text)
    if len(t) < n:
        return {t.encode("utf-8")}
    return {t[i : i + n].encode("utf-8") for i in range(len(t) - n + 1)}


# --------- 2) stable 64-bit hash for each shingle ---------


def hash64(data: bytes) -> int:
    # 64-bit stable hash
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)


# --------- 3) MinHash core ---------

# 选一个足够大的素数做模数（常用 2^61-1，运算快且够用）
P = (1 << 61) - 1


def make_hash_params(k: int, seed: int = 42) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    params = []
    for _ in range(k):
        a = rng.randrange(1, P)  # a != 0
        b = rng.randrange(0, P)
        params.append((a, b))
    return params


def minhash_signature(shingles: Iterable[bytes], k: int = 128, seed: int = 42) -> List[int]:
    params = make_hash_params(k, seed=seed)
    # 初始化为无穷大
    sig = [P] * k

    for sh in shingles:
        x = hash64(sh) % P
        for i, (a, b) in enumerate(params):
            hv = (a * x + b) % P
            if hv < sig[i]:
                sig[i] = hv

    return sig


def minhash_jaccard_estimate(sig1: Sequence[int], sig2: Sequence[int]) -> float:
    assert len(sig1) == len(sig2)
    eq = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return eq / len(sig1)


def datasketch_minhash(shingles: Iterable[bytes], k: int = 128) -> MinHash:
    m = MinHash(num_perm=k)
    for sh in shingles:
        m.update(sh)  # 必须是 bytes
    return m


if __name__ == "__main__":
    t1 = "今天讲 MinHash 和 LSH，用来做模糊去重。"
    t2 = "今天聊 MinHash/LSH，用于做近重复去重。"

    shingles1 = char_ngrams(t1, n=5)
    shingles2 = char_ngrams(t2, n=5)

    sig1 = minhash_signature(shingles1, k=128, seed=42)
    sig2 = minhash_signature(shingles2, k=128, seed=42)

    est = minhash_jaccard_estimate(sig1, sig2)
    print("MinHash estimated Jaccard =", est)

    shingles1 = char_ngrams(t1, n=5)
    shingles2 = char_ngrams(t2, n=5)

    m1 = datasketch_minhash(shingles1, k=128)
    m2 = datasketch_minhash(shingles2, k=128)

    print("datasketch MinHash estimated Jaccard =", m1.jaccard(m2))
    print("signature length =", len(m1.hashvalues))
    """
    MinHash estimated Jaccard = 0.171875
    datasketch MinHash estimated Jaccard = 0.1484375
    signature length = 128
    """
