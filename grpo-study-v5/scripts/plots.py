"""Write portable SVGs directly from the frozen report, without plotting dependencies."""
from html import escape
import math


COLORS = ("#477fa9", "#9164a3", "#29846b")


def chart(path, title, ylabel, points, lines, intervals=(), guides=()):
    width, height = 900, 410
    left, right, top, bottom = 75, 865, 56, 323
    values = [v for _, ys, _ in lines for v in ys] + list(guides)
    values += [v for _, lo, hi in intervals for v in (lo, hi)]
    lower, upper = min(values), max(values)
    tick = max(.25, math.ceil((upper-lower)/5*4)/4)
    lower = math.floor(lower/tick)*tick
    upper = max(lower+tick, math.ceil(upper/tick)*tick)
    min_x, max_x = min(points), max(points)
    def x(value):
        return left + (right-left)*(value-min_x)/max(1, max_x-min_x)
    def y(value):
        return bottom - (bottom-top)*(value-lower)/(upper-lower)
    def label(x_pos, y_pos, value, extra=""):
        return f'<text x="{x_pos}" y="{y_pos}" {extra}>{escape(str(value))}</text>'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="system-ui,sans-serif" font-size="13" fill="#223140">',
             label(left, 28, title, 'font-size="20" font-weight="bold"'),
             label(left, 49, ylabel)]
    for i in range(round((upper-lower)/tick)+1):
        value = lower+i*tick
        row = y(value)
        parts += [f'<path d="M{left} {row}H{right}" stroke="#e2e9f0"/>',
                  label(left-8, row+4, f"{value:g}", 'text-anchor="end"')]
    for guide in guides:
        row = y(guide)
        parts.append(f'<path d="M{left} {row}H{right}" stroke="#8997a4" stroke-dasharray="4 4"/>')
    last_label = -100
    for step in points:
        pos = x(step)
        if pos-last_label >= 38 or step == max_x:
            parts.append(label(pos, bottom+19, step, 'text-anchor="middle"'))
            last_label = pos
    parts.append(label((left+right)/2, bottom+40, "Optimizer updates", 'text-anchor="middle"'))
    for step, lo, hi in intervals:
        pos = x(step)
        parts.append(f'<path d="M{pos-4} {y(lo)}h8M{pos} {y(lo)}V{y(hi)}M{pos-4} {y(hi)}h8" stroke="#202a36"/>')
    for index, (name, ys, color) in enumerate(lines):
        coords = " ".join(f"{x(step):.1f},{y(v):.1f}" for step, v in zip(points, ys, strict=True))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for step, value in zip(points, ys, strict=True):
            parts.append(f'<circle cx="{x(step):.1f}" cy="{y(value):.1f}" r="3" fill="{color}"/>')
        parts.append(label(left+index*155, height-13, name, f'fill="{color}"'))
    path.write_text("\n".join(parts + ["</g></svg>"]) + "\n")


def write_plots(dest, report):
    for split, data in report["splits"].items():
        points = data["points"]
        steps = [point["step"] for point in points]
        baseline = 100*points[0]["mean_accuracy"]
        lines = [(f"Seed {seed}", [100*p["seed_metrics"][i]["accuracy"] for p in points], COLORS[i])
                 for i, seed in enumerate(report["seeds"])]
        lines.append(("Three-seed mean", [100*p["mean_accuracy"] for p in points], "#202a36"))
        chart(dest / f"{split}-accuracy.svg", f"{split.upper()} accuracy by training step",
              "Accuracy (%)", steps, lines, guides=[baseline])
        delta = [(name, [v-baseline for v in ys], color) for name, ys, color in lines]
        intervals = [(p["step"], *p["vs_sft"]["ci95_pp"]) for p in points[1:]]
        chart(dest / f"{split}-gain.svg", f"{split.upper()} gain over SFT (paired 95% intervals)",
              "Gain (percentage points)", steps, delta, intervals, [0])
        marginal = data["marginal"]
        if marginal:
            chart(dest / f"{split}-marginal.svg", f"{split.upper()} gain per 320 updates (simultaneous intervals)",
                  "Gain (percentage points)", [p["end"] for p in marginal],
                  [("Three-seed mean", [p["delta_pp"] for p in marginal], "#202a36")],
                  [(p["end"], *p["simultaneous_ci95_pp"]) for p in marginal], [-1, 0, 1])
