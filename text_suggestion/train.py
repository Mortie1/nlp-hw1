from backend.text_processing import WhiteSpaceTokenizer, extract_message_body
from backend.models import TextSuggestion, NGramLanguageModel, WordCompletor
import pandas as pd
import pickle
from tqdm import tqdm
import time

N = 3
MIN_CNT = 10


if __name__ == "__main__":
    tqdm.pandas()

    print("start")
    t = time.time()

    data = pd.read_csv("data/emails.csv")
    print(f"read time: {time.time() - t}")
    t = time.time()

    data.message = data.message.progress_apply(extract_message_body)
    print(f"extract time: {time.time() - t}")
    t = time.time()

    tokenizer = WhiteSpaceTokenizer(data.message, min_cnt=MIN_CNT, n=N)
    print(f"train tokenizer time: {time.time() - t}")
    t = time.time()

    corpus = data.message.progress_apply(lambda x: tokenizer.encode(x)).tolist()
    print(f"create corpus time: {time.time() - t}")
    t = time.time()

    word_completor = WordCompletor(corpus, min_cnt=MIN_CNT)
    print(f"train word completor time: {time.time() - t}")
    t = time.time()

    n_gram_model = NGramLanguageModel(corpus=corpus, n=N)
    print(f"train n gram model time: {time.time() - t}")
    t = time.time()

    text_suggestion = TextSuggestion(word_completor, n_gram_model, tokenizer)
    print(f"train text suggestion time: {time.time() - t}")

    with open("backend/pkl_models/text_suggestion.pkl", "wb") as f:
        pickle.dump(text_suggestion, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("backend/pkl_models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
