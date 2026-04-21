from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase

normalizer = normalizers.Sequence([
    NFD(),
    StripAccents(),
    Lowercase()
])

normalizer.normalize_str("Hello how are you?")
tokenizer.normalizer = normalizer

