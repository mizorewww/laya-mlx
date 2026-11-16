"""Dependency-free language/script detection used to route between Laya checkpoints.

Routing only needs one decision: *is this English Latin text, or is it something the English
checkpoint cannot read?* Benchmarks on MASSIVE (14 languages) showed the English checkpoint
collapsing to near-random on non-Latin scripts (Hindi 0.100, Korean 0.103, Swahili 0.103,
Tamil 0.113 at 20 options, where random is 0.050), while holding up far better on Latin-script
languages (French 0.487, Spanish 0.480). So the signal that matters most is *script*, and the
secondary signal is whether Latin text is English.

Script detection is exact, and answers the same way on every supported interpreter: a letter is
decided from the text, never from the Unicode database the host happens to ship (`_is_letter`).
The Latin-script language guess is a stopword/diacritic heuristic and is explicitly best-effort:
pass an explicit model or `lang=` when you already know the language.
"""

import bisect
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from typing import Dict, Iterator, List, Optional, Union

# Unicode blocks that the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
_SCRIPT_RANGES = [
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("armenian", ((0x0530, 0x058F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    (
        "arabic",
        ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    ),
    ("devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("oriya", ((0x0B00, 0x0B7F),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("sinhala", ((0x0D80, 0x0DFF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("lao", ((0x0E80, 0x0EFF),)),
    ("tibetan", ((0x0F00, 0x0FFF),)),
    ("myanmar", ((0x1000, 0x109F),)),
    ("georgian", ((0x10A0, 0x10FF),)),
    ("ethiopic", ((0x1200, 0x137F),)),
    ("khmer", ((0x1780, 0x17FF),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF))),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
]

_FLATTENED_RANGES = sorted(
    ((lo, hi, script) for script, ranges in _SCRIPT_RANGES for lo, hi in ranges),
    key=lambda item: item[0],
)
_STARTS = [lo for lo, hi, script in _FLATTENED_RANGES]
_ENDS = [hi for lo, hi, script in _FLATTENED_RANGES]
_SCRIPTS = [script for lo, hi, script in _FLATTENED_RANGES]
_bisect_right = bisect.bisect_right

# Letters the running interpreter's Unicode database may not know (see `_is_letter`).
#
# "Letters assigned after Unicode 14.0, plus the scripts Unicode 14 introduced": every codepoint here
# is category ``L*`` in Unicode 16.0 and ``Cn`` (unassigned) in Unicode 14.0 -- except the Unicode 14
# scripts themselves, which Unicode 14.0 does know and Unicode 13.0 (Python 3.10) does not, so 3.10
# gets the same answer. Generated from the two databases, not typed by hand; the CJK extensions are
# whole blocks because every ideograph in them is a letter.
_POST_DATABASE_LETTER_RANGES = (
    (0x01C89, 0x01C8A),  # CYRILLIC CAPITAL LETTER TJE
    (0x0A7CB, 0x0A7CD),  # LATIN CAPITAL LETTER RAMS HORN
    (0x0A7DA, 0x0A7DC),  # LATIN CAPITAL LETTER LAMBDA
    (0x10570, 0x1057A),  # VITHKUQI (Unicode 14)
    (0x1057C, 0x1058A),
    (0x1058C, 0x10592),
    (0x10594, 0x10595),
    (0x10597, 0x105A1),
    (0x105A3, 0x105B1),
    (0x105B3, 0x105B9),
    (0x105BB, 0x105BC),
    (0x105C0, 0x105F3),  # TODHRI (Unicode 16)
    (0x10D4A, 0x10D65),  # GARAY (Unicode 16)
    (0x10D6F, 0x10D85),
    (0x10EC2, 0x10EC4),  # ARABIC additions
    (0x10F70, 0x10F81),  # OLD UYGHUR (Unicode 14)
    (0x1123F, 0x11240),  # KHOJKI additions
    (0x11380, 0x11389),  # TULU-TIGALARI (Unicode 16)
    (0x1138B, 0x1138B),
    (0x1138E, 0x1138E),
    (0x11390, 0x113B5),
    (0x113B7, 0x113B7),
    (0x113D1, 0x113D1),
    (0x113D3, 0x113D3),
    (0x11BC0, 0x11BE0),  # SUNUWAR (Unicode 16)
    (0x11F02, 0x11F02),  # KAWI (Unicode 15)
    (0x11F04, 0x11F10),
    (0x11F12, 0x11F33),
    (0x12F90, 0x12FF0),  # CYPRO-MINOAN (Unicode 14)
    (0x1342F, 0x1342F),  # EGYPTIAN HIEROGLYPH additions
    (0x13441, 0x13446),
    (0x13460, 0x143FA),
    (0x16100, 0x1611D),  # GURUNG KHEMA (Unicode 16)
    (0x16A70, 0x16ABE),  # TANGSA (Unicode 14)
    (0x16D40, 0x16D6C),  # KIRAT RAI (Unicode 16)
    (0x18CFF, 0x18CFF),  # KHITAN SMALL SCRIPT addition
    (0x1B132, 0x1B132),  # HIRAGANA / KATAKANA additions
    (0x1B155, 0x1B155),
    (0x1DF25, 0x1DF2A),  # LATIN SMALL LETTER D WITH MID-HEIGHT LEFT HOOK
    (0x1E030, 0x1E06D),  # MODIFIER LETTER additions
    (0x1E290, 0x1E2AD),  # TOTO (Unicode 14)
    (0x1E4D0, 0x1E4EB),  # NAG MUNDARI (Unicode 15)
    (0x1E5D0, 0x1E5ED),  # OL ONAL (Unicode 16)
    (0x1E5F0, 0x1E5F0),
    (0x2B739, 0x2B739),  # CJK UNIFIED IDEOGRAPH extensions H and I
    (0x2EBF0, 0x2EE5D),
    (0x31350, 0x323AF),
)
# Cheapest rejection for a character the table could match: everything below the first range start is
# never consulted (the ASCII and Latin text almost every request carries), so the common path stays
# `str.isalpha()` plus one integer compare.
_POST_DATABASE_LETTER_FLOOR = _POST_DATABASE_LETTER_RANGES[0][0]


def _is_letter(ch: str) -> bool:
    """Whether `ch` is a letter, as the *newest* interpreter would say, not just this one.

    ``str.isalpha()`` and ``unicodedata.category`` are properties of the interpreter's bundled
    Unicode database, and the supported versions disagree: Python 3.10 ships Unicode 13.0, 3.11 14.0,
    3.12 15.0. A letter added since then is ``Cn`` -- *unassigned* -- on the older interpreter, so
    `isalpha()` is False and it was dropped before `_script_counts`' "other" branch could count it.
    Text written in one of those scripts therefore reported ``script="unknown"``, which is "no letters
    detected", and `Router._route` reads that as the default checkpoint rather than as non-Latin text
    -- the exact failure the "other" branch exists to prevent, and it made `analyse`/`is_english`
    answer differently for the same input depending on the interpreter.

    The table only SUPPLIES the answer a newer database would have given: a character this interpreter
    already knows is decided by `isalpha()` alone, so behaviour on the newest interpreter is unchanged
    codepoint for codepoint. What is a letter never depends on the host any more, for the scripts
    listed above.
    """
    if ch.isalpha():
        return True
    if ord(ch) < _POST_DATABASE_LETTER_FLOOR:
        return False
    if unicodedata.category(ch) != "Cn":
        return False
    cp = ord(ch)
    for lo, hi in _POST_DATABASE_LETTER_RANGES:
        if lo <= cp <= hi:
            return True
    return False


# Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
# weighted and a margin is required before calling something non-English.
#
# The Romance lists (fr/es/pt/it) deliberately carry the *unaccented* function words as well as the
# accented ones. A state that lost its accents -- mail clients, ticket systems and any pipeline that
# normalises to ASCII strip them -- keeps no diacritic rate for the non-English signal to read, so
# `la`, `un`, `y`, `e`, `et`, `deux` and friends are the only evidence left. With a list of mostly
# accented words such a state produced one hit or none, fell under the two-hit margin below, and was
# handed to the English checkpoint as undecided-but-not-non-English text (see #172 and #54).
_STOP = {
    "en": {
        "the",
        "and",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "for",
        "with",
        "that",
        "this",
        "it",
        "you",
        "have",
        "has",
        "not",
        "but",
        "on",
        "at",
        "be",
        "as",
        "from",
        "will",
        "can",
        "would",
        "there",
        "their",
        "what",
        "which",
        "please",
        "we",
        "i",
        # Question words, auxiliaries, pronouns and connectives. Without them a plain request
        # such as "Why hasn't my card been delivered?" matched no word above, stayed undecided
        # and took the router default, the multilingual checkpoint. Each one, added alone,
        # moved none of the 101,650 non-English MASSIVE validation rows to the English
        # checkpoint. Words that did stay out: `over` (Dutch, Danish, Norwegian), `just` and
        # `under` (Swedish), `still` (Norwegian), `he` (Spanish, Afrikaans, Finnish), `had`
        # (Dutch), `or` and `cant` (Welsh), `us` (`us dollar`), and `about`, `up`, `out`
        # (English titles and loanwords inside other languages). `also` is German, so it
        # stays out too.
        # `doesn`, `isn` and the rest are what `_WORD` leaves of a contraction; `doesnt` is how
        # it is typed without the apostrophe.
        "how",
        "why",
        "when",
        "where",
        "who",
        "whom",
        "does",
        "done",
        "doing",
        "been",
        "being",
        "could",
        "should",
        "might",
        "must",
        "shall",
        "doesn",
        "didn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "haven",
        "hasn",
        "couldn",
        "wouldn",
        "won",
        "dont",
        "wont",
        "didnt",
        "doesnt",
        "isnt",
        "wasnt",
        "havent",
        "ive",
        "youre",
        "thats",
        "whats",
        "any",
        "some",
        "many",
        "much",
        "every",
        "each",
        "these",
        "those",
        "both",
        "few",
        "they",
        "them",
        "she",
        "him",
        "his",
        "our",
        "your",
        "yours",
        "into",
        "onto",
        "between",
        "after",
        "before",
        "through",
        "during",
        "without",
        "within",
        "down",
        "if",
        "nor",
        "than",
        "then",
        "because",
        "while",
        "until",
        "here",
        "only",
        "very",
        "too",
        "ever",
        "never",
        "always",
        "already",
        "yet",
    },
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "une",
        "est",
        "pour",
        "dans",
        "que",
        "qui",
        "avec",
        "sur",
        "pas",
        "plus",
        "nous",
        "vous",
        "être",
        "cette",
        "mais",
        "sont",
        "ont",
        "aux",
        "ce",
        "et",
        "du",
        "au",
        "ou",
        "je",
        "tu",
        "il",
        "elle",
        "ils",
        "elles",
        "mon",
        "ton",
        "ma",
        "ta",
        "sa",
        "mes",
        "tes",
        "ses",
        "ces",
        "deux",
        "trois",
        "très",
        "bien",
        "tout",
        "tous",
        "toute",
        "fait",
        "veux",
        "veut",
        "peux",
        "peut",
        "dois",
        "doit",
        "merci",
        "bonjour",
        "jour",
        "jours",
        "mois",
        "fois",
        "quand",
        "comment",
        "pourquoi",
        "alors",
        "donc",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "ist",
        "ein",
        "eine",
        "den",
        "dem",
        "nicht",
        "mit",
        "für",
        "auf",
        "von",
        "zu",
        "sich",
        "auch",
        "werden",
        "wurde",
        "haben",
        "sind",
        "oder",
        "aber",
        "ich",
        "wir",
        "mir",
        "mich",
        "dir",
        "dich",
        "uns",
        "mein",
        "meine",
        "meinen",
        "meinem",
        "meiner",
        "diese",
        "dieser",
        "diesen",
        "dieses",
        "einen",
        "einem",
        "einer",
        "wie",
        "wo",
        "wann",
        "welche",
        "im",
        "zum",
        "zur",
        "aus",
        "bei",
        "nach",
        "noch",
        "bitte",
        "heute",
        "jetzt",
        "kann",
        "kannst",
        "habe",
        "gibt",
        "wird",
        # shared with English on purpose: counted for English alone, they outvoted short German
        "in",
        "was",
    },
    "es": {
        "el",
        "los",
        "las",
        "que",
        "por",
        "con",
        "para",
        "una",
        "es",
        "se",
        "del",
        "como",
        "pero",
        "son",
        "está",
        "este",
        "esta",
        "todo",
        "más",
        "muy",
        "hay",
        "sus",
        # `de`/`en` are Spanish too, but they are common English tokens as well (`de facto`,
        # `en-US`, `en route`, `Rio de Janeiro`), and a state of those alone already carries
        # no English function word for the margin below to weigh them against, so they stay out.
        "la",
        "un",
        "y",
        "al",
        "lo",
        "le",
        "les",
        "su",
        "mi",
        "tu",
        "nos",
        "ni",
        "dos",
        "tres",
        "fue",
        "fueron",
        "ser",
        "tiene",
        "tienen",
        "tengo",
        "puede",
        "pueden",
        "quiero",
        "necesito",
        "hemos",
        "han",
        "sobre",
        "entre",
        "cuando",
        "donde",
        "porque",
        "aunque",
        "también",
        "ya",
        "eso",
        "esto",
        "esa",
        "ese",
        "nada",
        "algo",
        "aquí",
        "hoy",
        "gracias",
    },
    "pt": {
        "os",
        "as",
        "que",
        "em",
        "um",
        "uma",
        "para",
        "com",
        "não",
        "é",
        "se",
        "do",
        "da",
        "dos",
        "das",
        "mas",
        "são",
        "está",
        "este",
        "esta",
        "muito",
        "pelo",
        "pela",
        # `no` is Portuguese too, and among the most frequent words it has; it is also one of the
        # most frequent English words, so it stays out and short Portuguese states that lean on it
        # alone are left to the diacritic rate, as before.
        "o",
        "e",
        "na",
        "nas",
        "nos",
        "ao",
        "aos",
        "por",
        "foi",
        "era",
        "ser",
        "sou",
        "tem",
        "tenho",
        "pode",
        "podem",
        "quero",
        "preciso",
        "eu",
        "meu",
        "minha",
        "seu",
        "sua",
        "isso",
        "isto",
        "aqui",
        "ali",
        "como",
        "quando",
        "onde",
        "porque",
        "mais",
        "já",
        "ainda",
        "agora",
        "hoje",
        "ontem",
        "dois",
        "três",
        "tudo",
        "nada",
        "obrigado",
        "olá",
        # Brazilian support text: `você` and the unaccented `nao`/`voce`/`sao`/`ja` that a stripped
        # state keeps (#172), and the chat abbreviations `vc`/`pra`. Without them "Voce pode me
        # mandar a nota fiscal?" matched one word and went to the English checkpoint, which on
        # `pt` reports 0.97 mean confidence at 0.47 accuracy. `ate`, `bom`, `sim` and `cade` stay
        # out: each is an English token too (ate, BOM, SIM, Cade).
        "você",
        "vocês",
        "voce",
        "voces",
        "vc",
        "vcs",
        "nao",
        "sao",
        "ja",
        "até",
        "tá",
        "pra",
        "gostaria",
        "obrigada",
        "também",
        "tambem",
        "estou",
        "estamos",
        "meus",
        "minhas",
        "nosso",
        "nossa",
        "consigo",
        "cadê",
        "boa",
        "tarde",
        "noite",
        # the words a ticket keeps once the jargon is English ("Deu erro 500 no endpoint de login
        # depois do update"): time and person words plus the past tenses a bug report is told in
        "depois",
        "antes",
        "então",
        "entao",
        "ninguém",
        "ninguem",
        "alguém",
        "alguem",
        "nenhum",
        "nenhuma",
        "estava",
        "ficou",
        "fiz",
        "deu",
    },
    "it": {
        "il",
        "lo",
        "gli",
        "che",
        "di",
        "per",
        "con",
        "non",
        "è",
        "si",
        "del",
        "della",
        "sono",
        "questo",
        "questa",
        "anche",
        "come",
        "più",
        "sono",
        "nella",
        "alla",
        "la",
        "le",
        "un",
        "uno",
        "una",
        "e",
        "ed",
        "o",
        "da",
        "su",
        "tra",
        "fra",
        "mi",
        "ci",
        "ne",
        "ho",
        "hai",
        "ha",
        "abbiamo",
        "avete",
        "hanno",
        "era",
        "stato",
        "stata",
        "devo",
        "deve",
        "devono",
        "voglio",
        "vorrei",
        "mio",
        "mia",
        "tuo",
        "sua",
        "quando",
        "dove",
        "perche",
        "molto",
        "poco",
        "sempre",
        "mai",
        "già",
        "ancora",
        "adesso",
        "oggi",
        "ieri",
        "grazie",
        "ciao",
        "scusa",
        # the articulated prepositions: Italian-only words, which is what lets a state made of
        # shared articles (`la fattura`) still name the language rather than stay undecided
        "nel",
        "nell",
        "negli",
        "sul",
        "sulla",
        "sulle",
        "dal",
        "dalla",
        "dallo",
        "dagli",
        "dei",
        "delle",
        "dello",
        "degli",
        "agli",
        "alle",
        "col",
    },
    "nl": {
        "het",
        "een",
        "van",
        "is",
        "op",
        "te",
        "dat",
        "niet",
        "met",
        "voor",
        "zijn",
        "aan",
        "door",
        "maar",
        "ook",
        "worden",
        "deze",
        "naar",
        "wordt",
    },
    # Swedish function words and common auxiliaries. Several overlap with English or German
    # (`i`, `kan`, `har`), so the distinctive words below are what lets Swedish text survive an
    # ASCII-normalising ticket pipeline without treating one stray Nordic letter as the only hint.
    "sv": {
        "jag",
        "är",
        "och",
        "inte",
        "att",
        "från",
        "till",
        "behöver",
        "får",
        "skulle",
        "ska",
        "vill",
        "måste",
        "också",
        "dessa",
        "detta",
        "säger",
        "upp",
        "utan",
        "mitt",
        "min",
        "om",
        "kommer",
        "här",
        "två",
        "vi",
        "nästa",
        "gör",
        "göra",
        "hjälp",
        "hjälpa",
        "mig",
        "återbetalning",
        "återbetala",
        "faktura",
        "gång",
        "gånger",
        "hittar",
        "inställningen",
        "inställningarna",
        "lösenord",
        "när",
        "öppnar",
        "spårningen",
        # Common spellings from ticket systems that strip Swedish diacritics.
        "aterbetalning",
        "aterbetala",
        "behover",
        "fel",
        "ganger",
        "hjalp",
        "hjalpa",
        "installningen",
        "installningarna",
        "kraschar",
        "kvittot",
        "losenord",
        "nar",
        "oppnar",
        "paket",
        "skicka",
        "sparningen",
        "tva",
        "uppdaterats",
        "blivit",
        "debiterade",
        "appen",
    },
    # Romanian words that its Romance neighbours do not share, so adding `ro` cannot steal a
    # French/Spanish/Italian/Portuguese state: `la`, `o`, `un`, `de`, `pe`, `ca` are deliberately
    # left out for that reason, and the diacritic signal below carries the rest.
    "ro": {
        "și",
        "să",
        "este",
        "sunt",
        "care",
        "pentru",
        "din",
        "dar",
        "după",
        "până",
        "fără",
        "ale",
        "lui",
        "în",
        "fost",
        "acum",
        "vreau",
        "trebuie",
        "foarte",
        "acest",
        "această",
        "acesta",
        "aceasta",
        "mi",
        "ți",
        "vă",
        "nu",
    },
    # Romanized Bangla ("Banglish"): how Bangla is typed in chats, tickets and email when no Bengali
    # keyboard is at hand. It has no diacritics, so without a list it read as undecided-but-English
    # and went to the English checkpoint, which scores 0.08 on Bangla at 0.94 confidence. Spelling
    # is not standardised, so the common variants are listed (`bhalo`/`valo`, `korchi`/`korsi`).
    # Left out on purpose: frequent Bangla words that are also English words -- `ache` (is),
    # `are` (is there), `to` (so), `take` (to him), `age` (before), `pore` (later), `mane`
    # (meaning), `din` (give), `sob` (all), `tar` (his), `dao` (give), `eta` (this; ETA) --
    # ordinary words of a neighbouring language (`ora`, `nei`, `vai`), and words another list
    # already claims (`na`, `o`, `e`, `je`, `ta`, `por`, `hoy`), so adding `bn` cannot move a
    # state of any other language.
    "bn": {
        "ami",
        "amar",
        "amake",
        "amra",
        "amader",
        "apni",
        "apnar",
        "apnake",
        "apnara",
        "tumi",
        "tomar",
        "tomake",
        "tomra",
        "tader",
        "ota",
        "eita",
        "oita",
        "ekta",
        "ei",
        "oi",
        "ki",
        "keno",
        "kivabe",
        "kibhabe",
        "kothay",
        "kokhon",
        "kobe",
        "koto",
        "kintu",
        "jodi",
        "tahole",
        "ar",
        "theke",
        "jonno",
        "sathe",
        "shathe",
        "diye",
        "niye",
        "moddhe",
        "kore",
        "korte",
        "korchi",
        "korsi",
        "korbo",
        "korechi",
        "koreche",
        "korun",
        "koren",
        "korlam",
        "hobe",
        "hoyeche",
        "hoise",
        "hocche",
        "hoyni",
        "chai",
        "chaina",
        "lagbe",
        "parchi",
        "parbo",
        "parchina",
        "peyechi",
        "paini",
        "dite",
        "dilam",
        "diyechi",
        "nai",
        "khub",
        "onek",
        "ekhon",
        "akhon",
        "ekhono",
        "abar",
        "ekbar",
        "duibar",
        "ajke",
        "kalke",
        "taka",
        "bhalo",
        "valo",
        "kharap",
        "shomossa",
        "somossa",
        "dhonnobad",
        "bhai",
        "shob",
        "keu",
        "kichu",
        "bolte",
        "bolun",
        "parben",
        "asbe",
        "jabe",
        "pabo",
        "ferot",
        "dorkar",
        "hoye",
        "geche",
        "gese",
    },
    # "her", "ne", "men", "de" collide with English, French and Romanian, and "ki" with the
    # romanized Bangla list, so they are left out.
    "az": {
        "və",
        "ve",
        "bir",
        "bu",
        "üçün",
        "ucun",
        "ilə",
        "ile",
        "olan",
        "olub",
        "olmasa",
        "var",
        "yox",
        "yoxdur",
        "mən",
        "sən",
        "biz",
        "siz",
        "onlar",
        "daha",
        "çox",
        "cox",
        "hər",
        "nə",
        "kimi",
        "görə",
        "sonra",
        "əgər",
        "eger",
        "deyil",
        "lakin",
        "amma",
        "ancaq",
        "artıq",
        "artiq",
        "də",
        "isə",
        "həm",
        "yalnız",
        "yalniz",
    },
}

# Short support fragments often consist of only two or three words, so they do not reach the
# four-token minimum used by the general language guess. These spellings are specific enough to
# identify Swedish in that narrow case; generic words such as `fel`, `hjälp`, `paket` and `appen`
# are deliberately not sufficient on their own. Keep ASCII-normalised variants beside the forms
# users commonly type without Swedish characters.
_SHORT_SWEDISH_WORDS = {
    "åtkomst",
    "atkomst",
    "lösenord",
    "losenord",
    "fakturan",
    "betalningen",
    "inloggningen",
    "glömt",
    "glomt",
    "behöver",
    "behover",
    "återbetalning",
    "aterbetalning",
    "kvitto",
    "kvittot",
    "spårningen",
    "sparningen",
    "inställningen",
    "installningen",
    "felmeddelande",
    "abonnemanget",
}
# Letters that ordinary English does not use. This is the signal that catches a Latin-script
# language we hold no stopwords for at all (Romanian, Polish, Czech, Turkish, Baltic, ...),
# which is the difference between routing it to the multilingual checkpoint and silently
# handing it to the English one.
_NON_EN_DIACRITICS = set(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"  # Western European
    "ăâîșțşţ"  # Romanian
    "ąćęłńśźż"  # Polish
    "čďěňřšťůž"  # Czech / Slovak
    "őű"  # Hungarian
    "ğı"  # Turkish (text is lowercased before matching)
    "āēģīķļņūž"  # Baltic
    "đ"  # Serbo-Croatian / Vietnamese
    "ə"  # Azerbaijani
    # Vietnamese: it holds no stopword list of its own, so its letters are the only
    # signal that keeps it off the English checkpoint. The set used to stop at `đ`
    # (the comment above already claimed Vietnamese coverage), but the tone-marked
    # letters -- `ả`, `ạ`, `ề`, `ơ`, `ư` and friends -- are precomposed codepoints
    # of their own, so everyday sentences such as "Cảm ơn bạn nhiều nhé" carried a
    # diacritic rate of 0.0 and were routed to the English checkpoint.
    # Text is lowercased before matching, so lowercase forms are enough.
    "ạảấầẩẫậắằẳẵặ"  # a-family tone marks
    "ẹẻẽếềểễệỉị"  # e-/i-family tone marks
    "ọỏốồổỗộớờởỡợụủứừửữự"  # o-/u-family tone marks
    "ỳỵỷỹơư"  # y-family tone marks, and the two letters
    # no other listed language uses
)
# Words that more than one list claims. `la`, `un`, `e`, `que`, `una` and friends are function words
# of several of these languages at once, so matching one says "not English" without saying *which*
# language: a shared word may not name a winner by itself (Romanian text was reported as French that
# way), though it still counts toward the total of a language that also matched a word of its own.
_SHARED_WORDS = {
    w
    for w in {word for words in _STOP.values() for word in words}
    if sum(w in words for words in _STOP.values()) > 1
}
# Danish is not in `_STOP`, but several of its common words also occur in the Swedish list.
# Do not let these words alone name a Danish sentence as Swedish; they remain useful score hits
# when another, more distinctive Swedish word is present.
_NORDIC_OVERLAP_WORDS = {
    "hej",
    "ja",
    "nej",
    "jo",
    "tack",
    "mig",
    "min",
    "om",
    "kommer",
    "får",
    "skulle",
    "vi",
}
_SHARED_WORDS.update(_NORDIC_OVERLAP_WORDS)
# English function words no other list holds (`in`, `is`, `as`, `was` are shared with German,
# Dutch and Portuguese). They alone carry the English rescue of `latin_profile`.
_EN_ONLY_WORDS = _STOP["en"] - _SHARED_WORDS
# A few of these function words are also ordinary English words: `come` (Italian), `son` (Spanish),
# `do` (Portuguese), `care` (Romanian), `todo` (a to-do list), `per`, `plus`, and `im` -- German
# `im`, and also how `I'm` is written without the apostrophe. For those a repeat is evidence of the
# foreign language exactly once: "do more, do less" repeats an English word, and counting both hits
# let one such word clear the `best >= 2` bar below and send plain English to the multilingual
# checkpoint. Every other word keeps counting occurrences: `der`, `los` and `sa` are nobody's
# English, so a request whose only evidence is `der` twice ("reduzieren der helligkeit der lichter")
# is still German, and deduping every word instead sent 791 of 148,700 non-English MASSIVE test rows
# to the English checkpoint. Dutch `van` is the word this list deliberately leaves out: "liedje van
# ... van" is ordinary Dutch, and deduping it moved 14 of those rows while rescuing no English one.
_EN_COLLISION_WORDS = {"come", "son", "do", "care", "todo", "im", "per", "plus"} & {
    w for lg, sw in _STOP.items() if lg != "en" for w in sw
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# A token whose dot or @ joins word characters is an identifier, not prose: `github.com`,
# `user@acme.com`, `v1.2.3`, `U.S.A.`. `_WORD` splits them into pieces that collide with real
# function words -- `com` is Portuguese for "with", `o` is its article, `e` is Italian "e" -- so a
# state that was mostly links scored a language it does not contain, and two domains were enough to
# cross the margin below. A sentence-final period (`arrivato.`) keeps its word: the pattern needs
# word characters on both sides of the dot.
# The lookbehind is what keeps this linear. Without it the greedy `[\w-]*` is retried
# at every offset inside a run of word characters, and each attempt rescans the run
# before failing on the absent `[.@]` -- quadratic in the run's length, so 4000
# characters of one token cost 205 ms against 0.45 ms for ordinary prose. It removes
# no match: `[\w-]` and `[.@]` are disjoint, so an attempt from inside a run consumes
# to exactly the same separator as an attempt from the run's start and the two always
# succeed or fail together -- a leftmost match can only ever begin at a run start.
_IDENTIFIER = re.compile(r"(?<![\w-])[\w-]*(?:[.@][\w-]+)+", re.UNICODE)


def _iter_text(
    state: Union[str, bytes, Mapping, list, None], _depth: int = 0, _paths=None, _path=()
) -> Iterator[str]:
    """Collect the string leaves of a state, so detection sees real content.

    Keys are ignored: they are usually English field names. Any mapping counts, not only
    `dict` -- a UserDict or mappingproxy has the same string values and used to contribute
    nothing, which made `analyse` report no letters and the router fall through to English.
    """
    if _depth > 6 or state is None:
        return
    if isinstance(state, str):
        if _paths is not None:
            _paths.append(_path)
        yield state
        return
    if isinstance(state, (bytes, bytearray)):
        try:
            text = bytes(state).decode("utf-8")
        except UnicodeDecodeError:
            return
        if _paths is not None:
            _paths.append(_path)
        yield text
        return
    if isinstance(state, Mapping):
        if _paths is None:
            for v in state.values():
                yield from _iter_text(v, _depth + 1)
        else:
            for key, v in state.items():
                # Do not stringify arbitrary keys or copy an unbounded field name into a report.
                bounded = (type(key) is str and len(key) <= 128) or (
                    type(key) is int and key.bit_length() <= 63
                )
                path = _path + (key,) if _path is not None and bounded else None
                yield from _iter_text(v, _depth + 1, _paths, path)
        return
    if isinstance(state, (list, tuple)):
        for i, v in enumerate(state):
            path = _path + (i,) if _paths is not None and _path is not None else _path
            yield from _iter_text(v, _depth + 1, _paths, path)
        return


def state_text(state: Union[str, bytes, Mapping, list, None], max_chars: int = 4000) -> str:
    """Flatten a state into the text used for detection (keys are ignored: they are usually English)."""
    if max_chars <= 0 or state is None:
        return ""
    if isinstance(state, str):
        return state if len(state) <= max_chars else state[:max_chars]
    if isinstance(state, (bytes, bytearray)):
        try:
            text = bytes(state).decode("utf-8")
        except UnicodeDecodeError:
            return ""
        return text if len(text) <= max_chars else text[:max_chars]
    if not isinstance(state, (Mapping, list, tuple)):
        return ""

    parts: List[str] = []
    budget = max_chars
    for leaf in _iter_text(state):
        if budget <= 0:
            break
        if len(leaf) > budget:
            parts.append(leaf[:budget])
            break
        parts.append(leaf)
        # Account for the joining space without materializing the full text first.
        budget -= len(leaf) + 1
    return " ".join(parts)[:max_chars]


def _script_counts(text: str) -> Dict[str, int]:
    """Count the alphabetic characters of `text` by script, in one pass.

    Latin is inserted last so `_script_from_counts` keeps `detect_script`'s tie-break: a named
    script wins a tie against Latin, because `max` returns the first of equal values and Latin
    is the last key. `analyse` needs both the dominant script and the per-script fractions, and
    used to walk the text twice (once per function) to get them; one pass serves both.
    """
    counts: Dict[str, int] = {}
    latin = 0
    for ch in text:
        if not _is_letter(ch):
            continue
        cp = ord(ch)
        if (
            cp < 0x02B0
            or 0x1E00 <= cp <= 0x1EFF
            or 0xFF21 <= cp <= 0xFF3A
            or 0xFF41 <= cp <= 0xFF5A
        ):
            latin += 1  # Latin, IPA Extensions, Ext-Additional, fullwidth
            continue
        idx = _bisect_right(_STARTS, cp) - 1
        if idx >= 0 and cp <= _ENDS[idx]:
            name = _SCRIPTS[idx]
            counts[name] = counts.get(name, 0) + 1
        else:
            # An alphabetic character no range claims used to be counted nowhere, so text
            # written only in an unlisted script produced a total of 0 and was reported as
            # "unknown" -- and `analyse` treats "unknown" as English, sending it to the
            # checkpoint that has no tokens for it. 68% of Unicode's alphabetic codepoints
            # are outside _SCRIPT_RANGES (the CJK extensions, kana supplements, bopomofo,
            # halfwidth katakana, and dozens of smaller scripts), so enumerating them all is
            # not maintainable. Counting the remainder under "other" keeps them visible and
            # non-Latin, which is the safe direction: an unreadable script must not be
            # handed to the English checkpoint.
            counts["other"] = counts.get("other", 0) + 1
    counts["latin"] = latin
    return counts


def _script_from_counts(counts: Dict[str, int]) -> str:
    if not any(counts.values()):
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _profile_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    # Keep Latin first in the public mapping, the order callers saw before this was refactored.
    ordered: Dict[str, int] = {}
    if counts.get("latin"):
        ordered["latin"] = counts["latin"]
    for name, value in counts.items():
        if name != "latin":
            ordered[name] = value
    return {k: v / total for k, v in ordered.items() if v}


def detect_script(text: str) -> str:
    """Dominant script of `text`: 'latin', 'han', 'devanagari', ... or 'unknown' if there are no letters."""
    return _script_from_counts(_script_counts(text))


def script_profile(text: str) -> Dict[str, float]:
    """Fraction of alphabetic characters belonging to each detected script."""
    return _profile_from_counts(_script_counts(text))


# A diacritic rate above this is taken as evidence the text is not English, even when no
# stopword list matches it.
NON_EN_DIACRITIC_RATE = 0.02

# One accented loanword or proper noun (`café`, `résumé`, `José`) must not alone pull otherwise
# plain English off the English checkpoint: the rate is measured over every character, so a
# single `é` in a short sentence clears the floor above. English function words keep their say
# only through the rescue below -- two distinct words no other list holds, at most one
# non-English letter word, and a rate still well above the floor vetoes regardless.
ENGLISH_RESCUE_DIACRITIC_RATE = 0.06

# Non-Latin text is not for the English checkpoint even when Latin letters are the plurality: a
# brand name or order code outvotes the CJK request around it letter for letter, though one CJK
# character carries far more than a letter. A short message needs a large share to count; a long
# payload (ticket fields, English agent turns) dilutes the share, so there a sentence's worth of
# letters counts too.
NON_LATIN_FRACTION = 0.2
NON_LATIN_MIN_FRACTION = 0.1
NON_LATIN_MIN_LETTERS = 10


def _non_latin_carries(non_latin: float, n_non_latin: int) -> bool:
    """Whether the non-Latin letters of a Latin-plurality text are enough to carry the state."""
    return non_latin >= NON_LATIN_FRACTION or (
        non_latin >= NON_LATIN_MIN_FRACTION and n_non_latin >= NON_LATIN_MIN_LETTERS
    )


def _script_of(ch: str) -> Optional[str]:
    """The named non-Latin script of one letter, or None for Latin and for unclaimed letters."""
    cp = ord(ch)
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return None
    idx = _bisect_right(_STARTS, cp) - 1
    if idx >= 0 and cp <= _ENDS[idx]:
        return _SCRIPTS[idx]
    return None


def _non_latin_words(text: str) -> List[str]:
    """Non-Latin runs that read as words rather than as annotation inside English prose.

    English prose carries three kinds of non-Latin letters that are not a request written in
    another script, and each is excluded here: a symbol (`Set α to 0.05`, one letter), a proper
    name (`Дмитрий Петрович Савицкий`, capitalised), and a pronunciation (`[vlɐˈdʲimʲɪr]`, which
    no script range claims). A combining mark belongs to the letter before it and never splits a
    word, so `Влади́мир` stays one capitalised name rather than becoming `Влади` + `мир`.
    """
    runs, cur, script = [], "", None
    for ch in text:
        if unicodedata.combining(ch):
            continue
        s = _script_of(ch)
        if s is not None and s == script:
            cur += ch
            continue
        if cur:
            runs.append(cur)
        cur, script = (ch, s) if s is not None else ("", None)
    if cur:
        runs.append(cur)
    return [w for w in runs if len(w) >= 2 and not w[0].isupper()]


def _english_rescued_by_words(words: List[str], diac_rate: float) -> bool:
    """Whether plain-English function words outvote a marginal diacritic rate (#337).

    The rate is measured over every character, so one accented loanword or proper noun in a
    short sentence clears `NON_EN_DIACRITIC_RATE` outright. English still wins when it shows at
    least two distinct function words no other list holds and at most one word carrying a
    non-English letter: one loanword is not a non-English vocabulary, while the odd word a
    Danish or Swedish sentence picks up (`i`, `at`, `for`, `have`) is not an English sentence
    either. A rate well above `ENGLISH_RESCUE_DIACRITIC_RATE` vetoes regardless.
    """
    if diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE:
        return False
    if len(set(words) & _EN_ONLY_WORDS) < 2:
        return False
    return sum(1 for w in set(words) if any(ch in _NON_EN_DIACRITICS for ch in w)) <= 1


def latin_profile(text: str) -> Dict[str, object]:
    """Evidence behind the Latin-script language guess.

    Returns `language` (may be None when undecided), `english_hits`, `diacritic_rate` and
    `looks_non_english`. The latter can be true for non-English diacritics or an overlapping
    Swedish-Danish marker even when this heuristic cannot name the language. `analyse` needs the
    evidence and not just the verdict, because "undecided" and "English" are different answers
    and only one of them is safe to send to the English checkpoint.

    A non-English language is only named when it matched at least one word that no other list
    claims: shared function words alone (`la`, `e`, `o`) identify no particular language.
    """
    return _latin_profile(text)


def _latin_profile(text: str, diagnostic=None) -> Dict[str, object]:
    # 'İ'.lower() is 'i' + a combining dot, which matches no word list
    words = _WORD.findall(_IDENTIFIER.sub(" ", text).replace("İ", "i").lower())
    lowered = text.lower()
    diac = sum(1 for ch in lowered if ch in _NON_EN_DIACRITICS)
    diac_rate = diac / max(1, len(lowered))
    non_english = diac_rate >= NON_EN_DIACRITIC_RATE
    nordic_overlap = bool(set(words) & _NORDIC_OVERLAP_WORDS) and not bool(
        set(words) & _EN_ONLY_WORDS
    )
    if 1 < len(words) < 4 and set(words) & _SHORT_SWEDISH_WORDS:
        if diagnostic is not None:
            diagnostic["reason"] = "short_swedish"
        return {
            "language": "sv",
            "english_hits": 0,
            "diacritic_rate": diac_rate,
            "looks_non_english": non_english,
        }
    if len(words) < 4:
        if diagnostic is not None:
            diagnostic["reason"] = "insufficient_words"
        return {
            "language": None,
            "english_hits": 0,
            "diacritic_rate": diac_rate,
            "looks_non_english": non_english or nordic_overlap,
        }

    # A collision word counts once however often it repeats; every other word counts its hits.
    counts = Counter(words)
    scores = {
        lg: sum(1 if w in _EN_COLLISION_WORDS else n for w, n in counts.items() if w in sw)
        for lg, sw in _STOP.items()
    }
    en = scores.get("en", 0)
    # Only a language that matched at least one word no other list claims may be named. Without
    # that condition the top score can be pure overlap -- `la` and `e` in Romanian text made
    # Italian the winner -- which is a guess dressed as a detection. Such a language is dropped
    # from the running rather than merely losing the tie, so a lesser score with real evidence
    # still gets named, and the text stays undecided when no list has any.
    evidenced = {
        lg: s
        for lg, s in scores.items()
        if lg != "en" and any(w not in _SHARED_WORDS for w in set(words) & _STOP[lg])
    }
    best_lg, best = max(evidenced.items(), key=lambda kv: kv[1], default=(None, 0))

    lang = None
    if best_lg and best >= max(2, en + 2):
        # a non-English language needs a clear margin over English function words
        lang = best_lg
    elif (
        best_lg == "sv"
        and "inte" in words
        and "kan" in words
        and words[0] in {"kan", "jag", "vi"}
        and en <= 1
    ):
        # Short login requests such as "kan inte logga in" carry a distinctive Swedish phrase
        # but also one English-shaped token (`in`). Do not treat `kan` alone as Swedish: it is
        # common in Danish and Norwegian too.
        lang = best_lg
    elif best_lg and non_english and best >= max(2, en):
        # Needs two hits here too. One shared function word ("para" in Turkish text) named Spanish
        # on the strength of the diacritics alone, which is a guess dressed as a detection.
        lang = best_lg
    elif en and (not non_english or _english_rescued_by_words(words, diac_rate)):
        lang = "en"
    if diagnostic is not None:
        diagnostic["reason"] = "language_evidence" if lang else "no_language_evidence"
    return {
        "language": lang,
        "english_hits": en,
        "diacritic_rate": diac_rate,
        "looks_non_english": non_english or (lang is None and nordic_overlap),
    }


def guess_latin_language(text: str) -> Optional[str]:
    """Best-effort language code for Latin-script text, or None when undecided.

    Scores function-word hits per language and requires the winner to beat English by a margin and
    to have matched at least one word of its own, so ordinary English is never misrouted and a
    word of several languages at once names none of them. Short inputs usually return None on
    purpose.
    """
    return latin_profile(text)["language"]


# Code is not prose in any language, but split into words it reads as one: `os.path` is Portuguese
# (`os`), `round(el, 2)` Spanish (`el`), `non_english` Italian (`non`). A line pasted from a program
# into an English request must not count as a foreign segment, so a line carrying code syntax --
# `=`, `;`, braces, brackets or a call `name(` -- is skipped, and dotted or underscored identifiers
# are dropped from the rest. Prose keeps "Deu erro (500)": the parenthesis follows a space.
_CODE_LINE = re.compile(r"[=;{}\[\]]|\w\(")
# Slash and backslash compounds are names, not sentences: `Nav/Com` and `OS/2` read as Portuguese
# (`com`, `os`), `C:\DOS\mode` as Portuguese (`dos`), `ESA/UN` as Spanish (`un`). A whitespace token
# holding a letter or digit, a joiner (`.`, `_`, `/`, `\`) and another letter or digit is an
# identifier or a compound and is dropped whole. The pattern has a fixed length on purpose: an
# open-ended `\w+(?:[._]\w+)+` backtracks quadratically on a long run of letters with no joiner,
# and a state is user input.
_JOINED = re.compile(r"[^\W_][._/\\][^\W_]")
# An all-caps token inside mixed-case text is an acronym or a code: `MON`, `LA`, `EST`, `COM`, `DES`
# are hockey teams, states, time zones and radio bands, not French or Portuguese. A segment written
# entirely in capitals keeps its words -- a customer shouting in Portuguese is still Portuguese.
_LETTER_RUN = re.compile(r"[^\W\d_]{2,}")
# ...but a line of nothing but acronyms has no lowercase either, so it skips the blanking above
# and is read as prose. `MON DES EST LA` -- a hockey team, a state, a time zone and an airport --
# was named French, and `STORES LOS ANGELES LAS VEGAS EL PASO CLOSED` Spanish. An all-caps line
# must therefore show evidence that is more than acronym-shaped before it is believed.
#
# The bar is on the tokens that actually scored, not on the line. Requiring one long word
# anywhere does almost nothing by itself: `ANGELES`, `VEGAS` and `SEGUNDO` are long and are not
# what named Spanish -- `LOS`, `LAS` and `EL` did.
#
# Three corpora decide these two numbers, and `research/evals/shouted_bar_corpora.py` builds all
# three and prints the table below: upper-cased MASSIVE test rows that the rule names as their
# own locale with no bar at all (genuine shouted prose, must survive); 500,000 lines of 4-8
# acronyms; 50,000 all-caps US address and signage lines. Pinned to the MASSIVE test split at
# revision 940fd47a and `random.Random(0)`, with the acronym and place-name pools committed in
# that script, so every row here can be rechecked when someone wants to move a constant. Each
# row applies one bar, named in full, so the rows are comparable:
#
#   rule                                   prose kept      acronym FP   address FP
#   none (no bar at all)                 10,960 100.00%      4,219         141
#   diacritics only                      10,960 100.00%      4,219         141
#   word >= 5 only                       10,854  99.03%        431         141  <- addresses stand
#   stopword >= 3 only                   10,729  97.89%      3,751         109  <- keeps LOS/LAS/EL
#   stopword >= 4 only                    8,841  80.67%          0           0
#   stopword >= 5 only                    6,963  63.53%          0           0
#   stopword >= 3 AND word >= 5          10,625  96.94%        386         109
#   stopword >= 4 AND word >= 5           8,762  79.95%          0           0  <- chosen
#   stopword >= 5 AND word >= 5           6,902  62.97%          0           0
#   stopword >= 4 AND word >= 6           8,418  76.81%          0           0
#   stopword >= 4 AND word >= 7           7,459  68.06%          0           0
#   stopword >= 4 AND word >= 5, no dia   6,749  61.58%          0           0
#
# `LOS`, `LAS` and `EL` are three, three and two letters, so any bar of 4 or more rejects them;
# what the table has to decide is the other side of that, and it is 3 that it rules out -- at 3
# the reported address line still reads as Spanish, and 3,751 of 500,000 acronym runs and 109 of
# 50,000 address lines still read as prose. 4 closes both columns outright. Going on to 5 buys
# nothing measurable and costs 1,860 more real lines. Non-English diacritics are accepted in
# place of a long stopword, because the module already treats them as non-English evidence
# (NON_EN_DIACRITIC_RATE) and neither an acronym nor a US place name carries any: that disjunct
# is free on both false-positive corpora (0 and 0 either way) and lifts prose kept from 61.58%
# to 79.95%.
_SHOUTED_MIN_STOPWORD = 4
# A length bar as well: a token of this many letters is a word rather than an acronym. Used
# twice -- here, on an all-caps line, and in `_analyse_text` to decide whether an all-caps run
# inside mixed-case text is an acronym worth blanking.
#
# Stated plainly, this bar is not justified by the table above: at a stopword bar of 4 both
# false-positive columns are already 0, so the 79 real prose lines it costs (8,841 -> 8,762) buy
# nothing those corpora can see. It is powerful on its own -- row three takes acronym false
# positives from 4,219 to 431 -- and it is kept because the corpora it was first chosen against
# were larger than the ones committed here (5,670 acronyms and 18,718 place names, neither
# committed, so their counts are not reproducible and are not quoted) and did report false
# positives surviving a stopword bar of 4. Absence on this corpus is not absence. Anyone who
# wants those 79 lines back should widen the acronym pool in that script first and show the
# column stays at 0. Swept with the stopword bar at 4: no bar 8,841 prose, >= 5 8,762,
# >= 6 8,418, >= 7 7,459.
#
# What the pair costs is not small and is not only short lines: 2,198 of the 10,960
# shouted-prose lines are given up, 1,816 of them (82.62%) six tokens or longer, the longest a
# 34-token Dutch sentence. On MASSIVE as written none of this is visible -- not one of the
# 151,674 rows holds a single uppercase character -- so it has to be measured on rows that do:
# over the 77,324 rows of the 26 non-English Latin locales, upper-cased, the multilingual share
# falls 0.5912 -> 0.5430, which is 3,727 rows. The casualties are sentences.
_SHOUTED_MIN_WORD = 5


def _shouted(text: str) -> bool:
    """True for a *cased* segment written entirely in capitals.

    The `isupper` half is not redundant. A caseless script -- Devanagari, Bengali, CJK -- has no
    lowercase either, so testing only `islower` would call every such segment shouted and hold it
    to a bar it cannot clear: `_WORD` splits at every combining mark, so the tokens are short by
    construction. Nothing is routed on that path today (`_STOP["bn"]` is romanised, so a
    Bengali-script line is named by script and never reaches here), but the bar belongs to cased
    text and saying so here keeps it that way.

    That argument covers *pure* caseless text, which never reaches a caps bar at all. Text that
    merely contains some has no lowercase either and does reach one; `_analyse_text` keeps both
    bars off it, because a bar that works by discarding Latin evidence has nothing to say about
    the half of the state it cannot read.
    """
    return any(ch.isupper() for ch in text) and not any(ch.islower() for ch in text)


def _shouted_evidence(tokens, lang: str, diacritic_rate: float) -> bool:
    """Whether an all-caps line's evidence for `lang` is more than acronym-shaped tokens."""
    if not any(len(t) >= _SHOUTED_MIN_WORD for t in tokens):
        return False
    if diacritic_rate >= NON_EN_DIACRITIC_RATE:
        return True
    matched = {t.lower() for t in tokens} & _STOP.get(lang, set())
    return any(len(w) >= _SHOUTED_MIN_STOPWORD for w in matched)


def _named_prose_language(segment: str):
    """Language code for one non-code line, or None when it does not name a foreign language.

    Same evidence bar as `_non_english_segment`: four words, a language `latin_profile` will name,
    and two *different* words of that language. Acronyms and slash compounds are not words. A
    segment in all capitals keeps its acronym-shaped tokens -- shouting is not a foreign language
    -- so it must also clear `_shouted_evidence`.
    """
    if not segment.strip() or _CODE_LINE.search(segment):
        return None
    prose = " ".join(tok for tok in segment.split() if not _JOINED.search(tok))
    shouted = _shouted(prose)
    if not shouted:
        prose = _LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), prose)
    tokens = _WORD.findall(prose)
    if len(tokens) < 4:
        return None
    prof = latin_profile(prose)
    lang = prof["language"]
    if lang in (None, "en"):
        return None
    if shouted and not _shouted_evidence(tokens, lang, float(prof["diacritic_rate"])):
        return None
    if len({w.lower() for w in tokens} & _STOP.get(lang, set())) < 2:
        return None
    return lang


def _non_english_segment(
    state: Union[str, bytes, Mapping, list, None], max_chars: int = 4000, diagnostic=None
):
    """First line or field that, read on its own, is named a non-English language, else None.

    Returns (language, segment). A segment needs the evidence a whole state needs -- at least four
    words, and a language named by `latin_profile` -- and, because one line carries far less text
    than a state, two things more: the words that name the language must be two *different* ones
    (`COM ... COM` in an English radio listing is one word seen twice), and acronyms and slash
    compounds are not words. This adds no new way to call English text foreign; it only stops a
    longer English part from outvoting a foreign one. Reads at most `max_chars` characters in all.
    """
    seen = 0
    paths = [] if diagnostic is not None else None
    for index, leaf in enumerate(_iter_text(state, _paths=paths)):
        for line, seg in enumerate(leaf.split("\n"), 1):
            if seen >= max_chars:
                return None
            seg = seg[: max_chars - seen]
            seen += len(seg)
            lang = _named_prose_language(seg)
            if lang:
                if diagnostic is not None:
                    diagnostic.update(
                        stage="segment",
                        reason="named_segment",
                        source=_diagnostic_source(paths, index, line),
                    )
                return lang, seg.strip()
    return None


def _diagnostic_source(paths, index, line):
    path = paths[index]
    return {"leaf_index": index, "path": list(path) if path is not None else None, "line": line}


def _analyse_text(text: str, diagnostic=None) -> Dict[str, object]:
    """Detection result for one already-flattened string.

    Does not look for a foreign line inside mostly-English text. `analyse` does that, because it
    needs the original state and not only the joined window.
    """
    counts = _script_counts(text)
    prof = _profile_from_counts(counts)
    script = _script_from_counts(counts)
    non_latin = round(1.0 - prof.get("latin", 0.0), 4) if prof else 0.0
    n_non_latin = round(non_latin * sum(ch.isalpha() for ch in text))
    if script == "latin" and _non_latin_words(text) and _non_latin_carries(non_latin, n_non_latin):
        script = max((s for s in prof if s != "latin"), key=prof.get)
    if script == "unknown":
        if diagnostic is not None:
            diagnostic["reason"] = "no_letters"
        return {
            "script": "unknown",
            "script_profile": prof,
            "language": None,
            "is_english": True,
            "language_undecided": True,
            "diacritic_rate": 0.0,
            "non_latin_fraction": 0.0,
            "mixed_segment": None,
        }
    if script != "latin":
        if diagnostic is not None:
            diagnostic["reason"] = "non_latin_script"
        return {
            "script": script,
            "script_profile": prof,
            "language": None,
            "is_english": False,
            "language_undecided": True,
            "diacritic_rate": 0.0,
            "non_latin_fraction": non_latin,
            "mixed_segment": None,
        }
    prof_lat = _latin_profile(text, diagnostic)
    lang = prof_lat["language"]
    # The acronym bar belongs here too, not only in the segment scan. The scan runs only once a
    # state already reads English overall, so a state that is *nothing but* an acronym line --
    # or one diluted by fewer English lines than it takes to tip this verdict -- never reached it
    # and was named foreign outright: `MON DES EST LA` routed multilingual on its own, and so did
    # the same line under one or two lines of English. Vetoing here leaves the text undecided, so
    # `looks_non_english` still decides it: a shouted line with non-English diacritics is kept.
    # Neither bar applies when the state is carried by a caseless script. Both of them answer
    # the verdict by taking evidence *away* from the Latin letters, and `looks_non_english`,
    # which decides what is left, reads Latin diacritics only -- it cannot see the caseless half.
    # Nothing else catches the fall, either: the script promotion above needs
    # `_non_latin_words`, which drops any run starting with a capital, so a shouted mixed-script
    # line has no promotable word by construction. `ПРИВЕТ MON DES EST LA` is 35% Cyrillic
    # letters and was being sent to the English checkpoint on the strength of disbelieving `MON
    # DES EST LA`. Whatever the Latin part is worth, text this far from Latin is not English, so
    # there is nothing for either bar to buy here and a wrong language name costs nothing: the
    # checkpoint is chosen on `is_english`.
    if lang not in (None, "en") and not _non_latin_carries(non_latin, n_non_latin):
        if _shouted(text):
            if not _shouted_evidence(_WORD.findall(text), lang, float(prof_lat["diacritic_rate"])):
                lang = None
                if diagnostic is not None:
                    diagnostic["reason"] = "caps_veto"
        else:
            # Mixed-case text: the segment scan has always held that an acronym is not a word,
            # but the whole-state verdict never applied that rule, so the acronyms voted in it.
            # That is what let a short English state be outvoted -- one or two lines of English
            # above `MON DES EST LA` still read as French overall, and only at three did English
            # win the margin. Re-take the verdict without the all-caps runs. `looks_non_english`
            # and `diacritic_rate` stay measured on the original text, so a foreign word that
            # happens to be shouted cannot be blanked out of the diacritic safety net.
            #
            # Only *acronym-shaped* runs are blanked. Blanking every all-caps run deleted
            # emphasis capitals, which are ordinary in real prose and are usually on the words
            # that carry the sentence, so when the shouted words were the ones that named the
            # language the evidence simply went: `sag mir das HEUTIGE DATUM` and `quiero
            # cancelar mi PEDIDO POR FAVOR` both lost their verdict and routed english. It also
            # broke monotonicity -- the fully upper-cased form held the shouted bar, but adding
            # one lowercase English token moved the text to this branch, where it had no bar at
            # all. So a run is blanked only while every one of them is short enough to be an
            # acronym, by `_shouted_evidence`'s own first test: a run of `_SHOUTED_MIN_WORD`
            # letters is a word, and a word is not blanked out of its own sentence. Measured
            # over the 20,708 Latin-locale MASSIVE test rows, emphasis-capitalised two ways
            # (see the commit message): regressions 1,753 -> 293 and 3,835 -> 752.
            caps = [m.group() for m in _LETTER_RUN.finditer(text) if m.group().isupper()]
            if caps and not any(len(c) >= _SHOUTED_MIN_WORD for c in caps):
                blanked = _LETTER_RUN.sub(lambda m: " " if m.group().isupper() else m.group(), text)
                if blanked != text:
                    lang = latin_profile(blanked)["language"]
                    if diagnostic is not None:
                        diagnostic["reason"] = "caps_filtered"
    # Undecided is not English. Treating it as English sent every Latin-script language we hold no
    # stopwords for to the checkpoint that cannot read it, silently. When nothing identifies the
    # language, non-English letters or a shared Swedish-Danish marker can still prefer the
    # multilingual checkpoint; text with neither signal still goes to the English one.
    undecided = lang is None
    english = lang == "en" or (undecided and not prof_lat["looks_non_english"])
    if diagnostic is not None and undecided and not english:
        diagnostic["reason"] = "non_english_markers"
    return {
        "script": "latin",
        "script_profile": prof,
        "language": lang,
        "is_english": english,
        "language_undecided": undecided,
        "diacritic_rate": round(float(prof_lat["diacritic_rate"]), 4),
        "non_latin_fraction": non_latin,
        "mixed_segment": None,
    }


def _leaf_non_english(leaf: str, diagnostic=None) -> Optional[Dict[str, object]]:
    """A string value that is itself not safe for the English checkpoint, else None.

    A one-word name ("José") and a capitalised non-Latin name stay out: the same rules
    `latin_profile` and `_non_latin_words` already use, so a name field cannot pull an
    English ticket onto the multilingual checkpoint. Code lines, acronyms and slash compounds
    stay out too, matching `_non_english_segment`, so a pasted traceback is not a message.
    Each line is capped at 4000 characters; unlike the segment scan, a long earlier field
    does not consume the budget of the next one (#384).
    """
    best_n = -1
    best: Optional[Dict[str, object]] = None
    for line_number, line in enumerate(leaf.split("\n"), 1):
        # A line this short cannot be selected, so skip it before the slice, `strip`,
        # `_CODE_LINE` and a full `_analyse_text` pass. Each branch below needs four `_WORD`
        # tokens or NON_LATIN_MIN_LETTERS letters. `_WORD` matches maximal runs of letters and
        # letter-like numerals, so four tokens need three separators between them: seven
        # characters, and ten letters need ten. What makes seven safe rather than six is that
        # every one of those counts is taken on the raw line -- `İ` lowers to two code points, so
        # counting on `sample.lower()` would let a four-character line reach four tokens, which is
        # why `latin_profile` replaces `İ` before lowering.
        if len(line) < 7:
            continue
        sample = line[:4000]
        if not sample.strip() or _CODE_LINE.search(sample):
            continue
        det = _analyse_text(sample)
        if det["is_english"]:
            continue
        # A named language still has to survive `_named_prose_language`: acronyms and slash
        # compounds are not words. But a veto is not a verdict of English. Dropping the line here
        # skipped the diacritic branch below, which exists for exactly this -- text that carries
        # non-English letters and that no stopword list can name. `WIE SPÄT IST ES IN KÖLN`
        # was vetoed for having no long stopword and then thrown away, so a German field routed
        # english; now it falls through and its umlauts carry it. That is also why
        # `language_undecided` is no longer required below: a language named and then disbelieved
        # is in the same evidential position as one never named.
        #
        # The ASCII spelling `WIE SPAET IST ES IN KOELN` is a different case and is not fixed
        # here. It has no diacritics to fall through to, and its longest matched German stopword
        # is `wie`, three letters, so `_SHOUTED_MIN_STOPWORD` vetoes it and nothing recovers it:
        # genuine German on the English checkpoint, one of the 3,727 upper-cased casualties
        # counted above and not an example of what this branch buys.
        named = det["language"] not in (None, "en") and _named_prose_language(sample) is not None
        if not named:
            if det["script"] not in ("latin", "unknown"):
                if not (
                    _non_latin_words(sample)
                    and sum(ch.isalpha() for ch in sample) >= NON_LATIN_MIN_LETTERS
                ):
                    continue
            elif not (
                float(det["diacritic_rate"]) >= NON_EN_DIACRITIC_RATE
                and len(_WORD.findall(sample)) >= 4
            ):
                continue
        n_alpha = sum(ch.isalpha() for ch in sample)
        if n_alpha > best_n:
            best_n = n_alpha
            best = det
            if diagnostic is not None:
                diagnostic.update(
                    line=line_number,
                    reason=(
                        "named_value"
                        if named
                        else "non_latin_value"
                        if det["script"] not in ("latin", "unknown")
                        else "diacritic_value"
                    ),
                )
    return best


def analyse(
    state: Union[str, bytes, Mapping, list, None], *, explain: bool = False
) -> Dict[str, object]:
    """Full detection result for a state.

    Returns `script`, `script_profile`, `language` (best effort, may be None),
    `is_english`, `non_latin_fraction` and `mixed_segment`. `mixed_segment` names the line or field
    the segment scan itself singled out, else None: a state flipped non-English by the per-value
    read below is reported with `mixed_segment` still None, because that read keeps the offending
    value in its own local and never writes it back.

    String values are what get read. When a state has several of them, one non-English value is
    enough: joining every value into one window let a long English note fill the 4000 characters,
    or outvote a short German message, and that message was then sent to the English checkpoint
    (#384). The segment scan still stops at 4000 characters, which is what keeps a huge field
    cheap; a value it did not reach is read on its own afterwards.

    With `explain=True`, adds `diagnostic`: the selected analysis stage, its reason and, for a
    segment or value override, the source leaf index, path and 1-based line. Paths retain string
    keys (up to 128 characters), integers (up to 63 bits) and sequence indices. Other keys make
    the path None; the leaf index remains available. No input text is added to the diagnostic.
    The original fields and detection rules are unchanged. These are heuristic reasons, not
    calibrated confidence or the router's final model choice.
    """
    diagnostic = {"stage": "whole_state", "reason": None, "source": None} if explain else None
    result = _analyse_text(state_text(state), diagnostic)
    if diagnostic is not None:
        result["diagnostic"] = diagnostic
    if result["script"] == "latin" and result["is_english"]:
        # A Portuguese ticket with an English stack trace, error payload or form template reads as
        # English as a whole, because the English part is longer -- yet the part a question is about
        # is the customer's, and the English checkpoint cannot read it (0.97 confidence at 0.47
        # accuracy on `pt`). The cost is lopsided: English sent to multilingual loses a few points,
        # the reverse loses calibration. So a state that would go to English is checked line by line
        # and field by field.
        # a single line has no other part to be outvoted by, and was just read whole
        it = iter(_iter_text(state))
        first = next(it, None)
        if first is not None and ("\n" in first or next(it, None) is not None):
            found = _non_english_segment(state, diagnostic=diagnostic)
            if found:
                lang, mixed = found
                result = dict(result)
                result["language"] = lang
                result["is_english"] = False
                result["language_undecided"] = False
                result["mixed_segment"] = mixed
    # A plain string was just read whole. A structured state can still hide a message past the
    # segment cap, or in a script `latin_profile` does not name.
    if isinstance(state, (str, bytes, bytearray)) or state is None or not result["is_english"]:
        return result
    best_n = -1
    best: Optional[Dict[str, object]] = None
    paths = [] if explain else None
    best_source = None
    best_reason = None
    for index, leaf in enumerate(_iter_text(state, _paths=paths)):
        leaf_diagnostic = {} if explain else None
        det = _leaf_non_english(leaf, leaf_diagnostic)
        if det is None:
            continue
        n_alpha = sum(ch.isalpha() for ch in leaf[:4000])
        if n_alpha > best_n:
            best_n = n_alpha
            best = det
            if explain:
                best_source = _diagnostic_source(paths, index, leaf_diagnostic["line"])
                best_reason = leaf_diagnostic["reason"]
    if best is None:
        return result
    updated = dict(result)
    updated["language"] = best["language"]
    updated["is_english"] = False
    updated["language_undecided"] = best["language_undecided"]
    if diagnostic is not None:
        diagnostic.update(stage="value", reason=best_reason, source=best_source)
    return updated


def is_english(state: Union[str, bytes, Mapping, list, None]) -> bool:
    """True when the English checkpoint can be expected to read this state."""
    return bool(analyse(state)["is_english"])
