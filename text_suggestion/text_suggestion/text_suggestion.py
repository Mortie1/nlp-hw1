import reflex as rx
from backend.models import TextSuggestion, WhiteSpaceTokenizer
import pickle
from rxconfig import config
import regex as re
import gc

with open("backend/pkl_models/text_suggestion.pkl", "rb") as f:
    gc.disable()
    text_suggestion: TextSuggestion = pickle.load(f)
    gc.enable()

with open("backend/pkl_models/tokenizer.pkl", "rb") as f:
    gc.disable()
    tokenizer: WhiteSpaceTokenizer = pickle.load(f)
    gc.enable()


class State(rx.State):
    """The app state."""

    input: str = ""
    output: str = ""
    n: str = "1"
    processing: bool = False
    complete: bool = False

    def process_input(self, data):
        """Get next words for input."""
        self.processing, self.complete = True, False
        self.input = data
        yield
        self.output = re.sub(
            r'\s([?.!"](?:\s|$))',
            r"\1",
            " ".join(text_suggestion.suggest_text(self.input, n_words=int(self.n))[0]),
        )
        self.processing, self.complete = False, True


def index():
    return rx.center(
        rx.vstack(
            rx.heading("N-Gram Text Suggestion", font_size="1.5em"),
            rx.input(
                placeholder="Enter text here...",
                on_change=State.process_input,
                width="25em",
            ),
            rx.select(["1", "2", "3", "4", "5"], value=State.n, on_change=State.set_n),
            rx.cond(
                State.complete,
                rx.text(State.output),
            ),
            align="center",
        ),
        width="100%",
        height="100vh",
    )


app = rx.App()
app.add_page(index, title="Text Suggestion")
