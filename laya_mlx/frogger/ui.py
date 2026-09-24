"""A fixed-cell terminal composition matching the Snake demo layout."""

from laya_mlx.snake.ui import AMBER, BG, CYAN, DIM, FG, GREEN, MUTED, RED, Canvas

from .game import DIRECTIONS, LEVEL_CROSSINGS

__all__ = ["BG", "compose", "layout_size"]

WATER = "#12384f"
LOG = "#9a6b3c"
ROAD = "#1a2327"
CAR_COLORS = (RED, AMBER, CYAN)


def layout_size(width, height):
    return max(104, width * 2 + 50), max(35, height + 19)


def compose(game, decision, stats):
    """Probabilities describe the displayed board, before its announced next step."""
    width, height = layout_size(game["width"], game["height"])
    c = Canvas(width, height)
    left, right, top = 3, max(58, game["width"] * 2 + 10), 6
    side = width - right - 4
    bottom = top + game["height"] + 1
    state = "PAUSED" if stats.get("paused") else "GAME OVER" if not game["alive"] else "LIVE"
    if stats.get("replay") and state == "LIVE":
        state = "RECORDED RUN · 1×"
    c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED)
    c.put(1, width - len(state) - 3, state, GREEN if game["alive"] else RED)
    c.put(2, left, "─" * (width - 6), DIM)
    c.put(4, left, "F R O G G E R", FG)
    c.put(4, left + 31, f"ROUND {stats.get('round', 1):02d}", MUTED)
    c.put(top, left, "┌" + "─" * (game["width"] * 2) + "┐", DIM)
    c.put(bottom, left, "└" + "─" * (game["width"] * 2) + "┘", DIM)
    for y in range(game["height"]):
        c.put(top + 1 + y, left, "│", DIM)
        c.put(top + 1 + y, left + game["width"] * 2 + 1, "│", DIM)
        if y == 0:
            c.put(top + 1 + y, left + 1, "◇ " * game["width"], "#2d6b52")
        elif 1 <= y <= 6:
            c.put(top + 1 + y, left + 1, "≈ " * game["width"], WATER)
            for x in game["logs"][y]:
                c.put(top + 1 + y, left + 1 + 2 * x, "██", LOG)
        elif y in (7, game["height"] - 1):
            c.put(top + 1 + y, left + 1, "· " * game["width"], "#13272e")
        else:
            c.put(top + 1 + y, left + 1, "· " * game["width"], ROAD)
            for x in game["cars"][y]:
                c.put(top + 1 + y, left + 1 + 2 * x, "██", CAR_COLORS[y % 3])
    x, y = game["frog"]
    c.put(top + y + 1, left + 1 + 2 * x, "██", GREEN if game["alive"] else RED)
    level = 1 + game["score"] // LEVEL_CROSSINGS
    for offset, label, value, color in (
        (0, "SCORE", game["score"], GREEN),
        (18, "LEVEL", level, FG),
        (36, "BEST", stats.get("best", game["score"]), MUTED),
    ):
        c.put(bottom + 2, left + offset, label, MUTED)
        c.number(bottom + 3, left + offset, value, color)
    progress = (game["height"] - 1 - y) / (game["height"] - 1)
    c.bar(bottom + 7, left, progress, 41)
    c.put(bottom + 7, left + 43, f"{100 * progress:4.1f}%", MUTED)

    c.put(4, right, "Laya MLX", GREEN)
    c.put(5, right, f"{stats.get('hardware', 'Apple silicon')} · Local", MUTED)
    c.put(7, right, "NEXT MOVE", FG)
    c.put(7, right + 15, "MODEL PROBABILITIES", MUTED)
    probabilities = decision.get("probabilities", {})
    for index, direction in enumerate(DIRECTIONS):
        row = 9 + index
        probability = probabilities.get(direction, 0)
        selected = direction == decision.get("proposed")
        color = GREEN if selected else MUTED
        c.put(row, right, f"{'›' if selected else ' '} {direction:<5}", color)
        count = round(probability * 18)
        c.put(row, right + 9, "░" * 18, DIM)
        c.put(row, right + 9, "█" * count, color)
        c.put(row, right + 29, f"{probability:.2f}", color)
    c.put(15, right, "EXECUTING", MUTED)
    c.put(15, right + 12, decision.get("executed", "—"), GREEN)
    if decision.get("intervened"):
        c.put(15, right + 20, "SHIELD", AMBER)
    c.put(17, right, "CRASH RISK", MUTED)
    risk = decision.get("crash_risk", 0)
    c.bar(18, right, risk, min(24, side - 9), AMBER if risk < 0.5 else RED)
    c.put(18, right + 29, f"{risk:.2f}", AMBER if risk < 0.5 else RED)
    c.put(20, right, "HOME REACHABLE", MUTED)
    c.bar(21, right, decision.get("home_reachable", 0), min(24, side - 9), CYAN)
    c.put(21, right + 29, f"{decision.get('home_reachable', 0):.2f}", CYAN)
    c.put(23, right, "INFERENCE", MUTED)
    c.put(23, right + 18, f"{decision.get('inference_ms', 0):5.1f} ms", FG)
    c.put(24, right, "DECISIONS", MUTED)
    c.put(24, right + 18, f"{stats.get('steps_per_second', 0):5.1f} /s", FG)
    c.put(25, right, "OUTPUT TOKENS", MUTED)
    c.put(25, right + 18, str(decision.get("output_tokens", 0)), FG)
    c.put(26, right, "NETWORK", MUTED)
    c.put(26, right + 18, "OFFLINE", GREEN)
    c.put(27, right, "ENGINE", MUTED)
    c.put(27, right + 18, "MLX · FP16", MUTED)
    guarded = stats.get("guarded", True)
    c.put(29, right, "Laya + lookahead safety" if guarded else "Laya · shield OFF", MUTED)
    c.put(30, right, f"Shield interventions  {stats.get('interventions', 0):04d}", AMBER)
    c.put(height - 3, left, "─" * (width - 6), DIM)
    c.put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", MUTED)
    elapsed = stats.get("elapsed", 0)
    clock = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
    c.put(height - 2, right, f"ESTIMATES BY LAYA            {clock}", MUTED)
    return c
