from typing import List, Optional
from collections import defaultdict, Counter
from tqdm import tqdm


def count_ngrams(
    corpus: List[List[str]], n: int = 3, min_prefix_cnt: int = 3, min_token_cnt: int = 3
):
    counts = defaultdict(Counter)

    for text in tqdm(corpus, desc="counting n-grams"):
        for i in range(n, len(text)):
            prefix = tuple(text[i - n : i])
            token = text[i]
            counts[prefix][token] += 1

    tokens_to_pop = []
    for prefix in list(counts.keys()):
        for token in list(counts[prefix].keys()):
            if counts[prefix][token] < min_token_cnt:
                tokens_to_pop.append((prefix, token))

    for prefix, token in tokens_to_pop:
        counts[prefix].pop(token)

    prefixes_to_pop = []
    for prefix in list(counts.keys()):
        if sum(counts[prefix].values()) < min_prefix_cnt:
            prefixes_to_pop.append(prefix)
    for prefix in prefixes_to_pop:
        counts.pop(prefix)
    return counts


class PrefixTreeNode:
    def __init__(self, parent: Optional["PrefixTreeNode"] = None, s: str = ""):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.word_children: List[PrefixTreeNode] = []
        self.parent: Optional[PrefixTreeNode] = parent
        self.s = s
        self.is_end_of_word = False

    def __repr__(self) -> str:
        return f"Node(\n\t{self.s=},\n\t{self.is_end_of_word=},\n\t{self.parent=},\n\t{self.children=},\n\t{self.word_children=}\n)"


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in tqdm(vocabulary, desc="building prefix tree"):
            cur_node = self.root
            for i, letter in enumerate(word):
                if not cur_node.children.get(letter):
                    cur_node.children[letter] = PrefixTreeNode(
                        cur_node, cur_node.s + letter
                    )
                cur_node = cur_node.children[letter]
                if i == len(word) - 1:
                    cur_node.is_end_of_word = True
                    word_to_save = cur_node.s
                    while cur_node != self.root:
                        cur_node.word_children += [word_to_save]
                        cur_node = cur_node.parent

    def search_prefix(self, prefix: str) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """
        cur_node = self.root
        for letter in prefix:
            cur_node = cur_node.children.get(letter)
            if not cur_node:
                return []
        return cur_node.word_children
        # your code here
