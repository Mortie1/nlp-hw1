import regex as re
from collections import Counter
from typing import List
from tqdm import tqdm


_empty_line_regex = re.compile(r"\n\s*\n")
_forwarded_regex = re.compile(
    r"-*\sForwarded by.*\s.*-*.*\n|(-)*(Original Message)(-)*\n"
)
_unstructured_meta_regex = re.compile(
    r"(^Forwarded by.*?$|^\s*\d+/\d+/\d{4}.*?$|^\s*From:.*?$|^\s*To:.*?$|^\s*cc:.*?$|^\s*Subject:.*?$)",
    flags=re.MULTILINE,
)
_unwanted_sequences_regex = re.compile(
    r"(\[IMAGE\]+|(=3D)+|(=20)+|(=\n)|([0-9]{3,})|([0-9]+)([\.\,])([0-9]*)|([0-9]+)((-([0-9]*))+)|=+|\$|\%|-{2,}|(https|http)?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}(\.[a-zA-Z0-9()]{1,6})?\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))|[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
    flags=re.MULTILINE,
)
_tabs_and_spaces_regex = re.compile(r"[ \t\n]+", flags=re.MULTILINE)


def extract_message_body(
    text: str,
) -> str:
    text = _empty_line_regex.split(text, 1)[-1].strip()
    text = _forwarded_regex.split(text, 1)[0]
    text = _unstructured_meta_regex.sub("", text)
    text = _unwanted_sequences_regex.sub("", text)
    text = _empty_line_regex.sub("\n", text).strip()
    text = _tabs_and_spaces_regex.sub(" ", text)
    return text.strip().lower()


class WhiteSpaceTokenizer:
    def __init__(self, corpus: List[str], min_cnt: int = 1, n: int = 2) -> None:
        self.n = n
        self.tokenizing_regex = re.compile(r"\w+|[^\w\s]+")

        words = []
        for text in tqdm(corpus, desc="getting words"):
            words.extend(self.tokenizing_regex.findall(text))

        word_count = Counter(words)
        if min_cnt != 1:
            for word in list(word_count.keys()):
                if word_count[word] < min_cnt:
                    word_count.pop(word)

        self.vocab = set(word_count.keys())
        self.vocab |= set(["[BOS]", "[EOS]", "[UNK]"])

    def encode(self, text: str) -> List[str]:
        words = self.tokenizing_regex.findall(text)
        return ["[BOS]"] * self.n + [
            word if word in self.vocab else "[UNK]" for word in words
        ]
