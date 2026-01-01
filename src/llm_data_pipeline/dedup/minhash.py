import re

import numpy as np
import xxhash

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


# --------- 2) Vectorized MinHash (NumPy) ---------


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
        # H shape: (N,) -> (N, 1)
        # self.a, self.b shape: (K,)
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
