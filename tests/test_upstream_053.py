"""v0.5.3 boundary regressions that run without native MLX or model weights."""

import pytest

from laya_mlx.lang import guess_latin_language, is_english
from laya_mlx.router import Router

SHORT_ASCII = [
    ("es", "no puedo entrar a mi cuenta"),
    ("es", "como hago para pedir un reembolso?"),
    ("es", "estoy esperando desde ayer una respuesta"),
    ("pt", "como faco para reembolsar?"),
    ("pt", "qual o prazo de entrega amanha?"),
]


@pytest.mark.parametrize(("lang", "text"), SHORT_ASCII)
def test_short_unaccented_spanish_and_portuguese_route_multilingual(lang, text):
    # Upstream 0.5.3 adds unaccented and closed-class stopwords so accent-stripped tickets are
    # recognized, alone and next to English boilerplate in a structured state.
    assert guess_latin_language(text) == lang
    assert is_english(text) is False
    assert Router(default="english").route(text).model == "multilingual"
    mixed = {"header": "Customer support ticket received from web form", "body": text}
    assert Router().route(mixed).model == "multilingual"


def test_italian_qual_stays_italian():
    assert guess_latin_language("qual è l'ultima notizia da rete quattro") == "it"


@pytest.mark.parametrize(
    "text",
    [
        "Please send me the MIS report before tomorrow morning",
        "I ate lunch and did not spend a dime on the trip",
        "Do me a favor and check the solo plan settings",
    ],
)
def test_english_controls_stay_english(text):
    assert is_english(text) is True
    assert Router().route(text).model == "english"
