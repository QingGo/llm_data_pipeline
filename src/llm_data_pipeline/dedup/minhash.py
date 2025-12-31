import random
import re
from collections.abc import Iterable, Sequence

import numpy as np
import xxhash
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
    # 64-bit stable hash using xxhash, which is much faster than blake2b for our use case
    return xxhash.xxh64_intdigest(data)


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
    def __init__(self, k: int = 128, seed: int = 42, ngram_size: int = 5):
        self.k = k
        self.seed = seed
        self.ngram_size = ngram_size
        # Using native uint64 arithmetic (modulo 2**64)
        self._init_perms()

    def _init_perms(self):
        """Initialize permutation parameters (a, b) for MinHash."""
        rng = np.random.RandomState(self.seed)
        
        # Generate k pairs of (a, b) over full uint64 range
        # a should be odd to ensure it's coprime with 2^64 (which is power of 2)
        
        # Efficiently generate full 64-bit random numbers
        # Use two 32-bit integers combined for better compatibility across numpy versions
        # This approach ensures we get full 64-bit entropy without issues on older numpy versions
        
        # Generate high and low 32-bit parts separately
        a_high = rng.randint(0, 2**32, size=self.k, dtype=np.uint64)
        a_low = rng.randint(0, 2**32, size=self.k, dtype=np.uint64)
        
        # Combine into 64-bit numbers and ensure a is odd
        self.a = (a_high << 32) | a_low
        self.a |= np.uint64(1)  # Ensure odd
        
        # Generate b similarly
        b_high = rng.randint(0, 2**32, size=self.k, dtype=np.uint64)
        b_low = rng.randint(0, 2**32, size=self.k, dtype=np.uint64)
        self.b = (b_high << 32) | b_low

    def compute_signature(self, text: str) -> list[int]:
        """Compute MinHash signature for a single text."""
        shingles = char_ngrams(text, n=self.ngram_size)
        
        # Handle empty text case - when text is empty, char_ngrams returns {b""}
        if len(shingles) == 1 and b"" in shingles:
            # Empty text case - return all zeros signature
            return [0] * self.k

        # 1. Generate stable hashes for all shingles
        # Using list comprehension for efficiency
        hashes = [xxhash.xxh64_intdigest(s) for s in shingles]
        
        # Early exit for texts with only one shingle
        if len(hashes) == 1:
            # Only one shingle, so just compute the permutations on it
            h = hashes[0]
            sigs = [(h * a + b) for a, b in zip(self.a, self.b, strict=True)]
            return sigs

        # 2. Convert to numpy array for vectorized computation
        H = np.array(hashes, dtype=np.uint64)
        
        # 3. Vectorized permutation and min calculation
        # Optimize memory usage by using smaller data types when possible
        # and leveraging numpy's built-in optimized functions
        
        # Reshape for broadcasting
        H_reshaped = H.reshape(-1, 1)
        
        # Compute all permutations in one vectorized operation
        # M shape: (N, K)
        M = H_reshaped * self.a + self.b
        
        # Compute minimum over each column (axis=0) to get the signature
        # M is already uint64, so result will be uint64
        sigs = np.min(M, axis=0)
        
        return sigs.tolist()
        
    def batch_compute_signature(self, texts: list[str]) -> list[list[int]]:
        """Compute MinHash signatures for multiple texts efficiently in batch.
        
        Args:
            texts: List of texts to compute signatures for
            
        Returns:
            List of MinHash signatures, one for each text
        """
        signatures = []
        for text in texts:
            sig = self.compute_signature(text)
            signatures.append(sig)
        return signatures


if __name__ == "__main__":
    t1 = "今天讲 MinHash 和 LSH，用来做模糊去重。"
    t2 = "今天聊 MinHash/LSH，用于做近重复去重。"

    # Check consistency
    vm = VectorizedMinHash(k=128, seed=42)
    sig1_v = vm.compute_signature(t1)

    shingles1 = char_ngrams(t1, n=5)
    sig1_legacy = minhash_signature(shingles1, k=128, seed=42)

    print(f"Vectorized matches Legacy? {sig1_v == sig1_legacy}")
