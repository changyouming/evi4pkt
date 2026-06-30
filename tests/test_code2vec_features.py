from __future__ import annotations

from evipkt.code2vec_features import (
    CODE2VEC_VECTOR_DIM,
    Code2VecVocab,
    build_vocab_from_codes,
    code2vec_vector,
    extract_ast_path_strings,
)


SAMPLE = """
public int sortaSum(int a, int b) {
    if (a + b >= 10 && a + b <= 19) {
        return 20;
    }
    return a + b;
}
"""


def test_extract_ast_paths_non_empty():
    paths = extract_ast_path_strings(SAMPLE, max_paths=20)
    assert paths
    assert all("," in p for p in paths)


def test_vocab_vector_dim():
    vocab = build_vocab_from_codes([SAMPLE], vocab_size=32, seed=0)
    vec = code2vec_vector(SAMPLE, vocab)
    assert len(vec) == CODE2VEC_VECTOR_DIM


def test_vocab_roundtrip(tmp_path):
    vocab = build_vocab_from_codes([SAMPLE], vocab_size=16, seed=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Code2VecVocab.load(path)
    assert loaded.path_to_index == vocab.path_to_index
    assert code2vec_vector(SAMPLE, loaded) == code2vec_vector(SAMPLE, vocab)
