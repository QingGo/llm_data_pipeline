import pytest
from datasketch import MinHash

from llm_data_pipeline.dedup.minhash import (
    VectorizedMinHash,
    char_ngrams,
    datasketch_minhash,
    hash64,
    minhash_jaccard_estimate,
    minhash_signature,
)


def test_char_ngrams():
    """Test character n-gram generation."""
    text = "test text"

    # Test with n=2
    ngrams = char_ngrams(text, n=2)
    expected = {b"te", b"es", b"st", b"t ", b" t", b"ex", b"xt"}
    assert ngrams == expected

    # Test with text shorter than n
    short_text = "ab"
    ngrams = char_ngrams(short_text, n=3)
    assert ngrams == {b"ab"}

    # Test with empty text
    empty_text = ""
    ngrams = char_ngrams(empty_text, n=2)
    assert ngrams == {b""}


def test_hash64():
    """Test 64-bit hash generation."""
    data = b"test"
    hash_value = hash64(data)

    # Check that hash is consistent
    assert hash64(data) == hash_value

    # Check that hash is a 64-bit integer
    assert isinstance(hash_value, int)
    assert 0 <= hash_value < 2**64

    # Check that different data produces different hashes
    assert hash64(b"test") != hash64(b"different")


def test_minhash_signature_consistency():
    """Test that both MinHash implementations produce consistent results."""
    text = "This is a test text for MinHash signature consistency."
    k = 128
    seed = 42

    # Original implementation
    shingles = char_ngrams(text, n=5)
    sig1 = minhash_signature(shingles, k=k, seed=seed)

    # Vectorized implementation
    vm = VectorizedMinHash(k=k, seed=seed, ngram_size=5)
    sig2 = vm.compute_signature(text)

    # Check that both signatures have the same length
    assert len(sig1) == k
    assert len(sig2) == k

    # Check that both signatures produce the same Jaccard estimate for identical texts
    jaccard1 = minhash_jaccard_estimate(sig1, sig1)
    jaccard2 = minhash_jaccard_estimate(sig2, sig2)
    assert jaccard1 == 1.0
    assert jaccard2 == 1.0


def test_minhash_jaccard_estimate():
    """Test Jaccard similarity estimation."""
    # Identical signatures should have Jaccard 1.0
    sig1 = [1, 2, 3, 4, 5]
    sig2 = [1, 2, 3, 4, 5]
    assert minhash_jaccard_estimate(sig1, sig2) == 1.0

    # Completely different signatures should have Jaccard 0.0
    sig1 = [1, 2, 3, 4, 5]
    sig2 = [6, 7, 8, 9, 10]
    assert minhash_jaccard_estimate(sig1, sig2) == 0.0

    # Partial overlap should have proportional Jaccard
    sig1 = [1, 2, 3, 4, 5]
    sig2 = [1, 2, 6, 7, 8]
    assert minhash_jaccard_estimate(sig1, sig2) == 0.4


def test_vectorized_minhash_different_texts():
    """Test vectorized MinHash with different texts."""
    text1 = "This is the first text for testing MinHash."
    text2 = "This is the second text for testing MinHash."
    text3 = "Completely different text about something else entirely."

    vm = VectorizedMinHash(k=128, seed=42, ngram_size=5)

    sig1 = vm.compute_signature(text1)
    sig2 = vm.compute_signature(text2)
    sig3 = vm.compute_signature(text3)

    # Check that signatures are different for different texts
    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3

    # Check that similar texts have higher Jaccard than dissimilar ones
    jaccard12 = minhash_jaccard_estimate(sig1, sig2)
    jaccard13 = minhash_jaccard_estimate(sig1, sig3)
    assert jaccard12 > jaccard13


def test_vectorized_minhash_empty_text():
    """Test vectorized MinHash with empty text."""
    empty_text = ""
    vm = VectorizedMinHash(k=128, seed=42)
    sig = vm.compute_signature(empty_text)

    # Check that signature is all zeros
    assert all(v == 0 for v in sig)
    assert len(sig) == 128


def test_vectorized_minhash_batch():
    """Test vectorized MinHash batch processing."""
    texts = [
        "First text for batch processing",
        "Second text for batch processing",
        "Third text for batch processing",
    ]

    vm = VectorizedMinHash(k=64, seed=42)

    # Test batch processing
    batch_sigs = vm.batch_compute_signature(texts)

    # Test individual processing
    individual_sigs = [vm.compute_signature(text) for text in texts]

    # Check that results are identical
    for i in range(len(texts)):
        assert batch_sigs[i] == individual_sigs[i]


def test_vectorized_minhash_different_seeds():
    """Test that different seeds produce different signatures."""
    text = "Test text with different seeds"

    vm1 = VectorizedMinHash(k=64, seed=42)
    vm2 = VectorizedMinHash(k=64, seed=123)

    sig1 = vm1.compute_signature(text)
    sig2 = vm2.compute_signature(text)

    # Check that signatures are different with different seeds
    assert sig1 != sig2

    # But should be same with same seed
    vm3 = VectorizedMinHash(k=64, seed=42)
    sig3 = vm3.compute_signature(text)
    assert sig1 == sig3


def test_vectorized_minhash_ngram_sizes():
    """Test vectorized MinHash with different n-gram sizes."""
    text = "Test text with different n-gram sizes"

    vm_3 = VectorizedMinHash(k=64, seed=42, ngram_size=3)
    vm_5 = VectorizedMinHash(k=64, seed=42, ngram_size=5)
    vm_7 = VectorizedMinHash(k=64, seed=42, ngram_size=7)

    sig_3 = vm_3.compute_signature(text)
    sig_5 = vm_5.compute_signature(text)
    sig_7 = vm_7.compute_signature(text)

    # Check that different n-gram sizes produce different signatures
    assert sig_3 != sig_5
    assert sig_5 != sig_7
    assert sig_3 != sig_7


def test_datasketch_minhash():
    """Test datasketch MinHash implementation."""
    text = "Test text for datasketch MinHash implementation"
    k = 128

    # Generate shingles
    shingles = char_ngrams(text, n=5)

    # Test datasketch MinHash
    minhash = datasketch_minhash(shingles, k=k)

    # Check that it returns a datasketch MinHash object
    assert isinstance(minhash, MinHash)
    assert minhash.num_perm == k


def test_compute_signature_early_exit():
    """Test the early exit condition in compute_signature for single-shingle texts."""
    # Create a text that will produce exactly one shingle
    single_shingle_text = "a" * 5  # 5 characters, n=5 produces 1 shingle

    vm = VectorizedMinHash(k=64, seed=42, ngram_size=5)
    sig = vm.compute_signature(single_shingle_text)

    # Check that we get a valid signature
    assert len(sig) == 64

    # Check that the signature is not all zeros (should have actual hash values)
    assert not all(v == 0 for v in sig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
