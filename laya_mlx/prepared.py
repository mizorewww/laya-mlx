"""Bounded tokenized-prefix reuse. Encoder states and predictions are never cached."""

from collections import OrderedDict
from dataclasses import dataclass
import threading

from .common import QTYPES, build_prefix, render_options, serialize_state


@dataclass(frozen=True)
class PreparedQuestion:
    ids: tuple
    markers: tuple


class PrefixCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.entries = OrderedDict()
        self._lock = threading.RLock()

    def prepare(self, agent, state, questions):
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        if not questions:
            return [], []
        tok = agent.tok
        max_len, head_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
        state_ids = tok(
            serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]
        items, internal = [], []
        for qid, definition in questions.items():
            q = agent._to_internal(definition)
            options = render_options(q)
            key = (
                id(tok),
                tok.cls_token_id,
                tok.sep_token_id,
                tok.mask_token_id,
                tok.mask_token,
                head_len,
                q["t"],
                q["ins"],
                tuple(options),
            )
            with self._lock:
                if key not in self.entries:
                    ids, markers = build_prefix(tok, q, head_len)
                    self.entries[key] = PreparedQuestion(tuple(ids), tuple(markers))
                    if len(self.entries) > self.capacity:
                        self.entries.popitem(last=False)
                self.entries.move_to_end(key)
                prefix = self.entries[key]
            room = max(0, max_len - len(prefix.ids) - 1)
            ids = (list(prefix.ids) + state_ids[:room] + [tok.sep_token_id])[:max_len]
            markers = [m for m in prefix.markers if m < max_len]
            if len(markers) != len(options):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal
