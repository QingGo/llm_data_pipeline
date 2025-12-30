import hashlib
import random
import re
from collections.abc import Iterable, Sequence

import numpy as np
from datasketch import MinHash

# --------- 1) normalize + shingling ---------


def normalize_text(text: str) -> str:
    # 你可以按需求加更多规则：Unicode NFKC、去HTML等
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngrams(text: str, n: int = 5) -> set[bytes]:
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


def make_hash_params(k: int, seed: int = 42) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    params = []
    for _ in range(k):
        a = rng.randrange(1, P)  # a != 0
        b = rng.randrange(0, P)
        params.append((a, b))
    return params


def minhash_signature(shingles: Iterable[bytes], k: int = 128, seed: int = 42) -> list[int]:
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
    eq = sum(1 for a, b in zip(sig1, sig2, strict=True) if a == b)
    return eq / len(sig1)


def datasketch_minhash(shingles: Iterable[bytes], k: int = 128) -> MinHash:
    m = MinHash(num_perm=k)
    for sh in shingles:
        m.update(sh)  # 必须是 bytes
    return m


# --------- 4) Vectorized MinHash (NumPy) ---------


class VectorizedMinHash:
    def __init__(self, k: int = 128, seed: int = 42):
        self.k = k
        self.seed = seed
        # Using native uint64 arithmetic (modulo 2**64)
        self._init_perms()

    def _init_perms(self):
        rng = np.random.RandomState(self.seed)
        # Generate k pairs of (a, b) over full uint64 range
        # We perform everything in uint64.
        # a should be odd to ensure it's coprime with 2^64 (which is power of 2)
        # numpy's randint with dtype=uint64 and large bounds can be tricky with legacy RandomState.
        # A simple way to get full 64-bit entropy:
        self.a = rng.randint(1, 2**62, size=self.k, dtype=np.uint64) * 4 + 1  # Ensure odd and spread
        # Actually proper way:
        # self.a = rng.bytes(self.k * 8) ...
        # But randint(1, 2**64...) might fail on some numpy versions.
        # Let's stick to safe large range that fits in int64 for generation, key is type is uint64 for calc.
        # Or just use two 32-bit generations combined.
        # For simplicity and speed let's just use high range available to randint.

        # Using 2**63-1 is safe for signed int64 inputs to randint, then cast to uint64
        a_base = rng.randint(1, 2**63 - 1, size=self.k, dtype=np.int64).astype(np.uint64)
        self.a = a_base | np.uint64(1)  # Ensure odd

        self.b = rng.randint(0, 2**63 - 1, size=self.k, dtype=np.int64).astype(np.uint64)

    def compute_signature(self, text: str) -> list[int]:
        shingles_set = char_ngrams(text, n=5)
        if not shingles_set:
            # Empty text case
            return [0] * self.k

        # 1. Provide stable hash for each shingle
        # map bytes -> u64
        hashes = [hash64(s) for s in shingles_set]

        # Convert to numpy array (N,)
        H = np.array(hashes, dtype=np.uint64)

        # 2. Vectorized Permutation
        # H shape: (N,)
        # a, b shape: (K,)
        # Broadcast: (N, 1) * (K,) + (K,) -> (N, K)
        # Native uint64 arithmetic wraps around 2^64 (modulo 2^64)

        # M = (H * a + b)
        # We need reshaping for broadcast

        # (N, 1) * (K,) -> (N, K)
        M = H.reshape(-1, 1) * self.a + self.b

        # 3. Min over columns (axis=0) -> (K,)
        sigs = M.min(axis=0)

        return sigs.tolist()


if __name__ == "__main__":
    t1 = "今天讲 MinHash 和 LSH，用来做模糊去重。"
    t2 = "今天聊 MinHash/LSH，用于做近重复去重。"

    # Check consistency
    vm = VectorizedMinHash(k=128, seed=42)
    sig1_v = vm.compute_signature(t1)

    shingles1 = char_ngrams(t1, n=5)
    sig1_legacy = minhash_signature(shingles1, k=128, seed=42)

    print(f"Vectorized matches Legacy? {sig1_v == sig1_legacy}")
