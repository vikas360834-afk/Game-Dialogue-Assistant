import math, pytest
from app.core import NGramModel, tokenize, detokenize, load_corpus, generate_dialogue, split_tokens

def test_preprocessing_and_corpus():
    assert tokenize("Hello, brave traveler!") == ["hello",",","brave","traveler","!"]
    assert len(load_corpus()) > 100

def test_add_one_probabilities():
    m=NGramModel(2).fit(["red","blue","red"])
    p=m.probabilities(["red"])
    assert all(x>0 for x in p.values()) and math.isclose(sum(p.values()),1)

def test_generation_repeatable_and_prompt():
    m=NGramModel(2).fit(tokenize("hello there friend . hello brave friend ."))
    assert m.generate("hello",8,42)==m.generate("hello",8,42)
    assert m.generate("hello",8,42).lower().startswith("hello")

def test_perplexity_finite():
    m=NGramModel(2).fit(["a","b","a","c"])
    assert math.isfinite(m.evaluate_perplexity(["a","b"]))

def test_validation():
    with pytest.raises(ValueError): NGramModel(1)
    with pytest.raises(ValueError): NGramModel().fit([])
    with pytest.raises(ValueError): NGramModel().fit(["x"]).generate(max_tokens=0)

def test_detokenize():
    assert detokenize(["hello",",","world","!"])=="Hello, world!"

def test_dialogue_options():
    m=NGramModel(2).fit(tokenize("hello friend . welcome traveler . hello brave friend ."))
    out=generate_dialogue(m,"hello",options=2,seed=5)
    assert len(out)==2 and all(x.lower().startswith("hello") for x in out)

def test_split():
    a,b=split_tokens(list("abcdefghij"),.8)
    assert len(a)==8 and len(b)==2
    with pytest.raises(ValueError): split_tokens(list("abc"),1)
