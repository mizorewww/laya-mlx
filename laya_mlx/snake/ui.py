"""A fixed-cell terminal composition shared by live display and recorded exports."""

from rich.text import Text

BG = "#090f13"
FG = "#e3f3ef"
MUTED = "#68868c"
DIM = "#20353c"
GREEN = "#62f5b5"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"

DIGITS = {
    "0": ("█▀█", "█ █", "▀▀▀"),
    "1": ("▄█ ", " █ ", "▀▀▀"),
    "2": ("▀▀█", "█▀▀", "▀▀▀"),
    "3": ("▀▀█", "▀▀█", "▀▀▀"),
    "4": ("█ █", "▀▀█", "  ▀"),
    "5": ("█▀▀", "▀▀█", "▀▀▀"),
    "6": ("█▀▀", "█▀█", "▀▀▀"),
    "7": ("▀▀█", "  █", "  ▀"),
    "8": ("█▀█", "█▀█", "▀▀▀"),
    "9": ("█▀█", "▀▀█", "▀▀▀"),
}


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.chars = [[" "] * width for _ in range(height)]
        self.styles = [[FG] * width for _ in range(height)]

    def put(self, row, column, text, color=FG):
        if not 0 <= row < self.height:
            return
        for offset, character in enumerate(str(text)):
            x = column + offset
            if 0 <= x < self.width:
                self.chars[row][x], self.styles[row][x] = character, color

    def bar(self, row, column, value, length=20, color=GREEN):
        count = round(max(0, min(1, value)) * length)
        self.put(row, column, "━" * length, DIM)
        self.put(row, column, "━" * count, color)

    def number(self, row, column, value, color=GREEN):
        digits = f"{value:03d}"
        for index, digit in enumerate(digits):
            for line, glyphs in enumerate(DIGITS[digit]):
                self.put(row + line, column + index * 4, glyphs, color)

    def rich_text(self):
        output = Text(no_wrap=True, overflow="crop", style=f"{FG} on {BG}")
        for row, (chars, styles) in enumerate(zip(self.chars, self.styles)):
            begin = 0
            for end in range(1, self.width + 1):
                if end == self.width or styles[end] != styles[begin]:
                    output.append("".join(chars[begin:end]), style=styles[begin])
                    begin = end
            if row + 1 != self.height:
                output.append("\n")
        return output


def layout_size(width, height):
    return max(104, width * 2 + 50), max(35, height + 19)


def compose(game, decision, stats):
    """Probabilities describe the displayed board, before its announced next step."""
    width, height = layout_size(game["width"], game["height"])
    c = Canvas(width, height)
    left, right, top = 3, max(58, game["width"] * 2 + 10), 6
    side = width - right - 4
    bottom = top + game["height"] + 1
    state = (
        "PAUSED"
        if stats.get("paused")
        else ("BOARD CLEAR" if game["won"] else "GAME OVER" if not game["alive"] else "LIVE")
    )
    if stats.get("replay") and state == "LIVE":
        state = "RECORDED RUN · 1×"
    c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED)
    c.put(1, width - len(state) - 3, state, GREEN if game["alive"] else RED)
    c.put(2, left, "─" * (width - 6), DIM)
    c.put(4, left, "S N A K E", FG)
    c.put(4, left + 31, f"ROUND {stats.get('round', 1):02d}", MUTED)
    c.put(top, left, "┌" + "─" * (game["width"] * 2) + "┐", DIM)
    c.put(bottom, left, "└" + "─" * (game["width"] * 2) + "┘", DIM)
    for y in range(game["height"]):
        c.put(top + 1 + y, left, "│", DIM)
        c.put(top + 1 + y, left + game["width"] * 2 + 1, "│", DIM)
        c.put(top + 1 + y, left + 1, "· " * game["width"], "#13272e")
    body = game["body"]
    for index, (x, y) in reversed(list(enumerate(body))):
        fraction = 1 - index / max(1, len(body))
        color = (
            "#dcfff0"
            if index == 0
            else (
                f"#{int(18 + 64 * fraction):02x}{int(73 + 150 * fraction):02x}"
                f"{int(57 + 102 * fraction):02x}"
            )
        )
        c.put(top + y + 1, left + 1 + 2 * x, "██", color)
    if game["food"] is not None:
        x, y = game["food"]
        c.put(top + y + 1, left + 1 + 2 * x, "● ", AMBER)
    for offset, label, value, color in (
        (0, "SCORE", game["score"], GREEN),
        (18, "LENGTH", game["length"], FG),
        (36, "BEST", stats.get("best", game["score"]), MUTED),
    ):
        c.put(bottom + 2, left + offset, label, MUTED)
        c.number(bottom + 3, left + offset, value, color)
    c.bar(bottom + 7, left, game["length"] / (game["width"] * game["height"]), 41)
    c.put(
        bottom + 7,
        left + 43,
        f"{100 * game['length'] / (game['width'] * game['height']):4.1f}%",
        MUTED,
    )

    c.put(4, right, "Laya MLX", GREEN)
    c.put(5, right, f"{stats.get('hardware', 'Apple silicon')} · Local", MUTED)
    c.put(7, right, "NEXT MOVE", FG)
    c.put(7, right + 15, "MODEL PROBABILITIES", MUTED)
    probabilities = decision.get("probabilities", {})
    for index, direction in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        row = 9 + index
        probability = probabilities.get(direction, 0)
        selected = direction == decision.get("proposed")
        color = GREEN if selected else MUTED
        c.put(row, right, f"{'›' if selected else ' '} {direction:<5}", color)
        count = round(probability * 18)
        c.put(row, right + 9, "░" * 18, DIM)
        c.put(row, right + 9, "█" * count, color)
        c.put(row, right + 29, f"{probability:.2f}", color)
    c.put(14, right, "EXECUTING", MUTED)
    c.put(14, right + 12, decision.get("executed", "—"), GREEN)
    if decision.get("intervened"):
        c.put(14, right + 20, "SHIELD", AMBER)
    c.put(16, right, "DEAD-END RISK", MUTED)
    risk = decision.get("dead_end_risk")
    if risk is None:
        c.put(17, right, "NOT COMPUTED / FAST MODE", MUTED)
    else:
        c.bar(17, right, risk, min(24, side - 9), AMBER if risk < 0.5 else RED)
        c.put(17, right + 29, f"{risk:.2f}", AMBER if risk < 0.5 else RED)
    c.put(19, right, "FOOD REACHABLE", MUTED)
    food = decision.get("food_reachable")
    if food is None:
        c.put(20, right, "NOT COMPUTED / FAST MODE", MUTED)
    else:
        c.bar(20, right, food, min(24, side - 9), CYAN)
        c.put(20, right + 29, f"{food:.2f}", CYAN)
    c.put(22, right, "INFERENCE", MUTED)
    c.put(22, right + 18, f"{decision.get('inference_ms', 0):5.1f} ms", FG)
    c.put(23, right, "DECISIONS", MUTED)
    c.put(23, right + 18, f"{stats.get('steps_per_second', 0):5.1f} /s", FG)
    c.put(24, right, "OUTPUT TOKENS", MUTED)
    c.put(24, right + 18, str(decision.get("output_tokens", 0)), FG)
    c.put(25, right, "NETWORK", MUTED)
    c.put(25, right + 18, "OFFLINE", GREEN)
    c.put(26, right, "ENGINE", MUTED)
    c.put(26, right + 18, "MLX · FP16", MUTED)
    guarded = stats.get("guarded", True)
    c.put(28, right, "Laya + cycle safety" if guarded else "Laya · shield OFF", MUTED)
    c.put(29, right, f"Shield interventions  {stats.get('interventions', 0):04d}", AMBER)
    c.put(height - 3, left, "─" * (width - 6), DIM)
    c.put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", MUTED)
    elapsed = stats.get("elapsed", 0)
    clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
    c.put(height - 2, right, f"ESTIMATES BY LAYA            {clock}", MUTED)
    return c
