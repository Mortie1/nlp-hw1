from collections import Counter
from typing import List
from typing import Union
from .utils import PrefixTree, count_ngrams
from .text_processing import WhiteSpaceTokenizer


class WordCompletor:

    def __init__(self, corpus, min_cnt=1):
        """
        corpus: list – корпус текстов
        """
        self.cnt = dict(Counter([word for text in corpus for word in text]))
        self.total_words = 0

        keys_to_pop = []
        for key in self.cnt.keys():
            if self.cnt[key] < min_cnt:
                keys_to_pop += [key]
            else:
                self.total_words += self.cnt[key]

        for key in keys_to_pop:
            self.cnt.pop(key)

        self.prefix_tree = PrefixTree(list(self.cnt.keys()))

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,

        с их вероятностями (нормировать ничего не нужно)
        """
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.cnt[word] / self.total_words for word in words]

        return words, probs


class NGramLanguageModel:
    def __init__(
        self,
        corpus: List[List[str]],
        n: int = 3,
        min_prefix_cnt: int = 3,
        min_token_cnt: int = 3,
    ):
        self.counts = count_ngrams(corpus, n, min_prefix_cnt, min_token_cnt)
        self.n = n

        self.prefix_counts = {
            prefix: sum(self.counts[prefix].values()) for prefix in self.counts
        }

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """

        if len(prefix) > self.n:
            prefix = prefix[-self.n :]

        prefix = tuple(prefix)
        possible_words = self.counts.get(prefix)
        if not possible_words:
            return ["[EOS]"], [1.0]

        next_words, probs = possible_words.keys(), [
            possible_words[token] / self.prefix_counts[prefix]
            for token in possible_words.keys()
        ]

        return next_words, probs


class TextSuggestion:
    def __init__(
        self,
        word_completor: WordCompletor,
        n_gram_model: NGramLanguageModel,
        tokenizer: WhiteSpaceTokenizer,
    ):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model
        self.tokenizer = tokenizer

    def suggest_text(
        self, text: Union[str, list], n_words=3, n_texts=1
    ) -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)

        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """

        suggestions = []

        if isinstance(text, str):
            # last_word = text.split(" ")[-1]
            # text = " ".join(text.split(" ")[:-1])
            text = self.tokenizer.encode(text)
            last_word = text[-1]
            text = text[:-1]
        else:
            last_word = text[-1]
            text = text[:-1]

        possible_last_words = list(
            zip(*self.word_completor.get_words_and_probs(last_word))
        )

        last_word_tokenized = "[UNK]"
        if last_word == "":
            pass
        elif not len(possible_last_words):
            suggestions += [last_word_tokenized]
        else:
            last_word_tokenized = max(
                filter(lambda x: x != "[UNK]", possible_last_words), key=lambda x: x[1]
            )[0]
            suggestions += [last_word_tokenized]

        for _ in range(n_words):
            suggestions += [
                max(
                    zip(
                        *self.n_gram_model.get_next_words_and_probs(text + suggestions)
                    ),
                    key=lambda x: x[1],
                )[0]
            ]

        if last_word_tokenized == "[UNK]" and last_word != "":
            suggestions = [last_word] + suggestions[1:]
        return [
            list(
                filter(
                    lambda x: x != "[EOS]" and x != "[BOS]" and x != "[UNK]",
                    suggestions,
                )
            )
        ]
