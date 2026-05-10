"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          gui.py  —  Graphical User Interface                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Sections:                                                           ║
║    1. Pipeline Runner  — run all 5 steps with live console output    ║
║    2. Keyword Extractor— extract keywords from any custom text       ║
║    3. Results Viewer   — browse charts + evaluation summary table    ║
║    4. About            — project info                                ║
║                                                                      ║
║  HOW TO RUN:                                                         ║
║    pip install pillow                                                ║
║    python gui.py                                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import re
import math
import queue
import threading
import subprocess
from collections import Counter

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ── Optional: Pillow for chart display ───────────────────────────────
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── Optional: NumPy + Gensim for live extraction ─────────────────────
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim.models import Word2Vec
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ── NLTK stopwords ───────────────────────────────────────────────────
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words('english'))
except Exception:
    STOP_WORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you',"you're",
        'your','yours','yourself','he','him','his','himself','she','her',
        'hers','it','its','itself','they','them','their','theirs','what',
        'which','who','whom','this','that','these','those','am','is','are',
        'was','were','be','been','being','have','has','had','having','do',
        'does','did','doing','a','an','the','and','but','if','or','because',
        'as','until','while','of','at','by','for','with','about','against',
        'between','into','through','during','before','after','above','below',
        'to','from','up','down','in','out','on','off','over','under','then',
        'once','here','there','when','where','why','how','all','both','each',
        'few','more','most','other','some','such','no','nor','not','only',
        'same','so','than','too','very','s','t','can','will','just','don',
        'should','now','d','ll','m','o','re','ve','y','ain','ma',
    }

from paths import (
    DATASET, GEN_DIR, EVAL_DIR,
    W2V_MODEL, TFIDF_FEATURES_NPY, EVAL_FIGS,
    EVAL_RESULTS_CSV, ALL_INTERMEDIATE, ALL_EVAL_OUTPUTS,
)

# ══════════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════════
BG_SIDE    = "#1a1f36"
BG_MAIN    = "#f0f2f8"
BG_CARD    = "#ffffff"
BG_CONSOLE = "#0d1117"
ACCENT     = "#4f6ef7"
ACCENT_HVR = "#3a57e8"
ACCENT2    = "#00c896"
TEXT_LIGHT = "#ffffff"
TEXT_DARK  = "#1e2140"
TEXT_DIM   = "#7b8eb8"
SUCCESS    = "#22c55e"
WARNING    = "#f59e0b"
ERROR      = "#ef4444"
BORDER     = "#dde3f0"
STEP_DONE  = "#22c55e"
STEP_RUN   = "#f59e0b"
STEP_IDLE  = "#94a3b8"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)
FONT_NAV    = ("Segoe UI", 11, "bold")
FONT_KW     = ("Segoe UI", 10, "bold")

PIPELINE_STEPS = [
    ("1", "Preprocessing",             "preprocessing.py"),
    ("2", "TF-IDF Extraction",         "TF_IDF.py"),
    ("3", "Word2Vec Embeddings",       "word_embeddings.py"),
    ("4", "Advanced Extraction",       "advanced_keyword_extraction.py"),
    ("5", "Evaluation & Comparison",   "evaluation.py"),
]

FIG_LABELS = {
    "fig1": "Summary Bar — P / R / F1",
    "fig2": "Per-Document F1",
    "fig3": "F1 Distribution (Boxplot)",
    "fig4": "Per-Doc TP / FP / FN",
    "fig5": "Total TP / FP / FN",
    "fig6": "Manual Evaluation Table",
    "fig7": "Radar Chart",
    "fig8": "F1 by Sentiment",
    "fig9": "Confusion Matrix",
}


# ══════════════════════════════════════════════════════════════════════
# HELPER — keyword extraction on raw text (no corpus needed)
# ══════════════════════════════════════════════════════════════════════
def preprocess_text(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r'<.*?>', ' ', text)
    text   = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return tokens


def tfidf_style_keywords(tokens: list, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Single-document keyword scoring:
      score(w) = TF(w) * log(1 + 1/TF(w))
    This rewards words that appear moderately (not too rare, not generic).
    Bigrams get a small bonus.
    """
    if not tokens:
        return []
    freq  = Counter(tokens)
    total = len(tokens)
    scored = {}
    for w, cnt in freq.items():
        tf = cnt / total
        scored[w] = tf * math.log(1.0 + 1.0 / tf)

    # Bigrams bonus
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    bg_freq = Counter(bigrams)
    for bg, cnt in bg_freq.items():
        if cnt > 1:
            scored[bg] = scored.get(bg, 0) + 0.3 * (cnt / total)

    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    # normalise to [0,1]
    if ranked:
        max_s = ranked[0][1]
        ranked = [(w, round(s / max_s, 4)) for w, s in ranked]
    return ranked[:top_n]


def w2v_style_keywords(tokens: list, model, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Word2Vec semantic extraction:
      score(w) = cosine_similarity(word_vec, doc_mean_vec)
    """
    if not tokens or model is None:
        return []
    known = [w for w in tokens if w in model.wv]
    if not known:
        return []
    vecs    = np.array([model.wv[w] for w in known])
    doc_vec = vecs.mean(axis=0, keepdims=True)
    sims    = cosine_similarity(doc_vec, vecs)[0]
    seen, out = set(), []
    for i in np.argsort(sims)[::-1]:
        w = known[i]
        if w not in seen:
            seen.add(w)
            out.append((w, round(float(sims[i]), 4)))
        if len(out) == top_n:
            break
    return out


# ══════════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ══════════════════════════════════════════════════════════════════════
def make_card(parent, **kwargs) -> tk.Frame:
    kw = dict(bg=BG_CARD, relief="flat", bd=0,
              highlightthickness=1, highlightbackground=BORDER)
    kw.update(kwargs)
    return tk.Frame(parent, **kw)


def make_btn(parent, text, command, bg=ACCENT, fg=TEXT_LIGHT,
             font=FONT_BODY, padx=18, pady=8, **kwargs) -> tk.Button:
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg=fg, font=font,
                    padx=padx, pady=pady,
                    relief="flat", cursor="hand2",
                    activebackground=ACCENT_HVR, activeforeground=TEXT_LIGHT,
                    bd=0, **kwargs)
    return btn


def make_label(parent, text, font=FONT_BODY, fg=TEXT_DARK, **kwargs) -> tk.Label:
    return tk.Label(parent, text=text, font=font, fg=fg,
                    bg=kwargs.pop("bg", BG_CARD), **kwargs)


# ══════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keyword Extraction System — Project 5")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=BG_SIDE)

        self._w2v_model  = None   # loaded lazily
        self._active_nav = tk.StringVar(value="pipeline")
        self._log_queue  = queue.Queue()
        self._step_status: dict[str, str] = {s[0]: "idle" for s in PIPELINE_STEPS}
        self._current_fig_idx = 0

        self._build_layout()
        self._show_section("pipeline")
        self.after(100, self._poll_log_queue)

    # ── Layout skeleton ───────────────────────────────────────────────
    def _build_layout(self):
        # Left sidebar
        self.sidebar = tk.Frame(self, bg=BG_SIDE, width=230)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Right content
        self.content = tk.Frame(self, bg=BG_MAIN)
        self.content.pack(side="left", fill="both", expand=True)

        self._build_sidebar()

        # One frame per section — only the active one is shown
        self.sections: dict[str, tk.Frame] = {}
        for name in ("pipeline", "extract", "results", "about"):
            f = tk.Frame(self.content, bg=BG_MAIN)
            self.sections[name] = f

        self._build_pipeline_section()
        self._build_extract_section()
        self._build_results_section()
        self._build_about_section()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var,
                 bg="#111827", fg=TEXT_DIM, font=FONT_SMALL,
                 anchor="w", padx=12).pack(side="bottom", fill="x")

    # ── Sidebar ───────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = self.sidebar

        # Logo area
        logo_frame = tk.Frame(sb, bg=BG_SIDE, pady=28)
        logo_frame.pack(fill="x")
        tk.Label(logo_frame, text="🔑", font=("Segoe UI", 32),
                 bg=BG_SIDE, fg=ACCENT).pack()
        tk.Label(logo_frame, text="Keyword\nExtraction",
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_SIDE, fg=TEXT_LIGHT, justify="center").pack()
        tk.Label(logo_frame, text="NLP Project 5",
                 font=FONT_SMALL, bg=BG_SIDE, fg=TEXT_DIM).pack()

        tk.Frame(sb, bg="#2d3555", height=1).pack(fill="x", padx=20)

        # Nav buttons
        nav_items = [
            ("pipeline", "⚙  Pipeline Runner"),
            ("extract",  "🔍  Keyword Extractor"),
            ("results",  "📊  Results Viewer"),
            ("about",    "ℹ  About"),
        ]
        self._nav_btns: dict[str, tk.Button] = {}
        for key, label in nav_items:
            btn = tk.Button(sb, text=label, font=FONT_NAV,
                            bg=BG_SIDE, fg=TEXT_DIM,
                            relief="flat", bd=0, padx=20, pady=14,
                            anchor="w", cursor="hand2",
                            activebackground="#252d4a",
                            command=lambda k=key: self._show_section(k))
            btn.pack(fill="x")
            self._nav_btns[key] = btn

        # Dataset status at bottom
        tk.Frame(sb, bg="#2d3555", height=1).pack(fill="x", padx=20, side="bottom", pady=4)
        self._ds_label = tk.Label(sb, text="", font=FONT_SMALL,
                                  bg=BG_SIDE, fg=TEXT_DIM, wraplength=180,
                                  justify="left", padx=14, pady=6)
        self._ds_label.pack(side="bottom", fill="x")
        self._refresh_ds_status()

    def _refresh_ds_status(self):
        if os.path.exists(DATASET):
            self._ds_label.config(text=f"✔ Dataset found\n{DATASET}", fg=SUCCESS)
        else:
            self._ds_label.config(text=f"✘ Dataset missing\nPlace {DATASET} here", fg=ERROR)

    def _show_section(self, name: str):
        for key, btn in self._nav_btns.items():
            if key == name:
                btn.config(bg="#252d4a", fg=TEXT_LIGHT)
            else:
                btn.config(bg=BG_SIDE, fg=TEXT_DIM)
        for key, frame in self.sections.items():
            if key == name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        self._active_nav.set(name)
        if name == "results":
            self._load_current_fig()
        self._refresh_ds_status()


    # ─────────────────────────────────────────────────────────────────
    # SECTION 1 — PIPELINE RUNNER
    # ─────────────────────────────────────────────────────────────────
    def _build_pipeline_section(self):
        parent = self.sections["pipeline"]

        # ── Header ───────────────────────────────────────────────────
        hdr = tk.Frame(parent, bg=BG_MAIN, pady=20, padx=30)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⚙  Pipeline Runner", font=FONT_TITLE,
                 bg=BG_MAIN, fg=TEXT_DARK).pack(side="left")
        make_btn(hdr, "▶  Run Full Pipeline", self._run_all_steps,
                 bg=ACCENT, font=("Segoe UI", 11, "bold"),
                 padx=22, pady=10).pack(side="right")

        # ── Step cards ───────────────────────────────────────────────
        steps_frame = tk.Frame(parent, bg=BG_MAIN, padx=30)
        steps_frame.pack(fill="x")

        self._step_cards: dict[str, dict] = {}
        for num, name, script in PIPELINE_STEPS:
            card = make_card(steps_frame, pady=10, padx=16)
            card.pack(fill="x", pady=4)

            # Left: indicator dot + text
            left = tk.Frame(card, bg=BG_CARD)
            left.pack(side="left", fill="x", expand=True)

            dot = tk.Label(left, text="●", font=("Segoe UI", 16),
                           bg=BG_CARD, fg=STEP_IDLE)
            dot.pack(side="left", padx=(0, 10))

            tk.Label(left, text=f"Step {num}", font=("Segoe UI", 9),
                     bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")
            title_lbl = tk.Label(left, text=name,
                                 font=("Segoe UI", 11, "bold"),
                                 bg=BG_CARD, fg=TEXT_DARK)
            title_lbl.pack(anchor="w")
            tk.Label(left, text=script, font=FONT_MONO,
                     bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

            # Right: run button + status label
            right = tk.Frame(card, bg=BG_CARD)
            right.pack(side="right")
            status_lbl = tk.Label(right, text="Idle", font=FONT_SMALL,
                                  bg=BG_CARD, fg=STEP_IDLE, width=8)
            status_lbl.pack()
            run_btn = make_btn(right, f"Run",
                               command=lambda n=num: self._run_step(n),
                               bg="#e8ecfb", fg=ACCENT,
                               font=("Segoe UI", 9, "bold"),
                               padx=14, pady=5)
            run_btn.pack(pady=4)

            self._step_cards[num] = {
                "dot": dot, "status": status_lbl,
                "btn": run_btn, "script": script
            }

        # ── Console ───────────────────────────────────────────────────
        console_frame = tk.Frame(parent, bg=BG_MAIN, padx=30, pady=6)
        console_frame.pack(fill="both", expand=True)

        tk.Label(console_frame, text="Console Output",
                 font=FONT_HEAD, bg=BG_MAIN, fg=TEXT_DARK).pack(anchor="w", pady=(4, 2))

        btn_row = tk.Frame(console_frame, bg=BG_MAIN)
        btn_row.pack(anchor="e")
        make_btn(btn_row, "Clear", self._clear_console,
                 bg="#e8ecfb", fg=ACCENT,
                 font=FONT_SMALL, padx=10, pady=4).pack(side="right")

        self.console = scrolledtext.ScrolledText(
            console_frame, bg=BG_CONSOLE, fg="#c9d1d9",
            font=FONT_MONO, wrap="word",
            insertbackground="white", state="disabled",
            relief="flat", bd=0)
        self.console.pack(fill="both", expand=True, pady=(2, 10))

        # colour tags
        for tag, color in [("ok",   SUCCESS), ("err", ERROR),
                            ("warn", WARNING), ("dim", TEXT_DIM),
                            ("head", ACCENT)]:
            self.console.tag_config(tag, foreground=color)

    def _clear_console(self):
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")

    def _log(self, msg: str, tag: str = ""):
        self._log_queue.put((msg, tag))

    def _poll_log_queue(self):
        try:
            while True:
                msg, tag = self._log_queue.get_nowait()
                self.console.config(state="normal")
                if tag:
                    self.console.insert("end", msg + "\n", tag)
                else:
                    self.console.insert("end", msg + "\n")
                self.console.see("end")
                self.console.config(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _set_step_status(self, num: str, status: str):
        colours = {"idle": STEP_IDLE, "running": STEP_RUN,
                   "done": STEP_DONE, "error": ERROR}
        labels  = {"idle": "Idle", "running": "Running…",
                   "done": "Done ✔", "error": "Failed ✘"}
        c = self.sections["pipeline"]
        card = self._step_cards[num]
        card["dot"].config(fg=colours.get(status, STEP_IDLE))
        card["status"].config(text=labels.get(status, ""), fg=colours.get(status, STEP_IDLE))
        self._step_status[num] = status

    def _run_step(self, num: str, _silent_start=False):
        card   = self._step_cards[num]
        script = card["script"]
        if not _silent_start:
            self._log(f"\n{'='*55}", "head")
            self._log(f"  Running Step {num}: {script}", "head")
            self._log(f"{'='*55}", "head")
        self._set_step_status(num, "running")
        self.status_var.set(f"Running Step {num} — {script}…")

        def worker():
            try:
                proc = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if any(x in line for x in ["ERROR", "FAILED", "Traceback"]):
                        self._log(line, "err")
                    elif any(x in line for x in ["Saved", "COMPLETE", "done", "✔"]):
                        self._log(line, "ok")
                    elif any(x in line for x in ["WARNING", "warn"]):
                        self._log(line, "warn")
                    else:
                        self._log(line)
                proc.wait()
                if proc.returncode == 0:
                    self.after(0, lambda: self._set_step_status(num, "done"))
                    self.after(0, lambda: self.status_var.set(f"Step {num} completed ✔"))
                else:
                    self.after(0, lambda: self._set_step_status(num, "error"))
                    self.after(0, lambda: self.status_var.set(f"Step {num} FAILED"))
            except Exception as e:
                self._log(str(e), "err")
                self.after(0, lambda: self._set_step_status(num, "error"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_all_steps(self):
        def sequencer():
            for num, _, script in PIPELINE_STEPS:
                if not os.path.exists(script):
                    self._log(f"[ERROR] {script} not found", "err")
                    self.after(0, lambda n=num: self._set_step_status(n, "error"))
                    return
                event = threading.Event()
                self.after(0, lambda n=num: self._run_step(n, _silent_start=False))
                # Poll until this step finishes
                while True:
                    import time; time.sleep(0.5)
                    if self._step_status.get(num) in ("done", "error"):
                        break
                if self._step_status.get(num) == "error":
                    self._log(f"\nPipeline stopped at Step {num}.", "err")
                    return
            self._log("\n✔ All steps completed!", "ok")
            self.after(0, lambda: self.status_var.set("Pipeline complete ✔"))
        threading.Thread(target=sequencer, daemon=True).start()


    # ─────────────────────────────────────────────────────────────────
    # SECTION 2 — KEYWORD EXTRACTOR
    # ─────────────────────────────────────────────────────────────────
    def _build_extract_section(self):
        parent = self.sections["extract"]

        # Header
        hdr = tk.Frame(parent, bg=BG_MAIN, pady=20, padx=30)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔍  Keyword Extractor", font=FONT_TITLE,
                 bg=BG_MAIN, fg=TEXT_DARK).pack(side="left")

        # Input card
        in_card = make_card(parent, padx=24, pady=18)
        in_card.pack(fill="x", padx=30, pady=(0, 10))

        tk.Label(in_card, text="Enter or paste your text below:",
                 font=FONT_HEAD, bg=BG_CARD, fg=TEXT_DARK).pack(anchor="w")
        tk.Label(in_card, text="The system will extract keywords using TF-IDF Style and Word2Vec Semantic methods.",
                 font=FONT_SMALL, bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", pady=(2, 8))

        self.text_input = tk.Text(in_card, height=7, font=FONT_BODY,
                                  bg="#f8faff", fg=TEXT_DARK, wrap="word",
                                  relief="flat", bd=0,
                                  highlightthickness=1, highlightbackground=BORDER,
                                  padx=10, pady=8, insertbackground=ACCENT)
        self.text_input.pack(fill="x")

        # Placeholder
        self._placeholder = "Paste any text here — article, abstract, movie review, news story…"
        self.text_input.insert("1.0", self._placeholder)
        self.text_input.config(fg=TEXT_DIM)
        self.text_input.bind("<FocusIn>",  self._on_text_focus_in)
        self.text_input.bind("<FocusOut>", self._on_text_focus_out)

        # Controls row
        ctrl = tk.Frame(in_card, bg=BG_CARD, pady=10)
        ctrl.pack(fill="x")

        tk.Label(ctrl, text="Keywords to extract:", font=FONT_BODY,
                 bg=BG_CARD, fg=TEXT_DARK).pack(side="left")
        self.top_n_var = tk.IntVar(value=10)
        spin = ttk.Spinbox(ctrl, from_=3, to=20, width=4,
                           textvariable=self.top_n_var, font=FONT_BODY)
        spin.pack(side="left", padx=(6, 20))

        make_btn(ctrl, "⚡  Extract Keywords",
                 self._extract_keywords,
                 bg=ACCENT, font=("Segoe UI", 10, "bold"),
                 padx=20, pady=8).pack(side="left")

        make_btn(ctrl, "Clear",
                 self._clear_extractor,
                 bg="#e8ecfb", fg=ACCENT,
                 font=FONT_SMALL, padx=12, pady=8).pack(side="left", padx=8)

        # W2V model status
        self._w2v_status = tk.Label(ctrl, text="", font=FONT_SMALL,
                                     bg=BG_CARD, fg=TEXT_DIM)
        self._w2v_status.pack(side="right")

        # Results area — two columns
        res_frame = tk.Frame(parent, bg=BG_MAIN, padx=30)
        res_frame.pack(fill="both", expand=True)

        res_frame.columnconfigure(0, weight=1)
        res_frame.columnconfigure(1, weight=1)

        self._kw_cols: dict[str, tk.Frame] = {}
        for col_idx, (key, title, color) in enumerate([
            ("tfidf",  "TF-IDF Style",         ACCENT),
            ("w2v",    "Word2Vec Semantic",     ACCENT2),
        ]):
            card = make_card(res_frame, padx=18, pady=14)
            card.grid(row=0, column=col_idx, sticky="nsew", padx=(0,10) if col_idx==0 else (0,0), pady=4)

            hd = tk.Frame(card, bg=BG_CARD)
            hd.pack(fill="x", pady=(0, 8))
            tk.Label(hd, text="●", font=("Segoe UI", 14),
                     bg=BG_CARD, fg=color).pack(side="left")
            tk.Label(hd, text=f"  {title}",
                     font=FONT_HEAD, bg=BG_CARD, fg=TEXT_DARK).pack(side="left")

            body = tk.Frame(card, bg=BG_CARD)
            body.pack(fill="both", expand=True)
            self._kw_cols[key] = body

        self._update_w2v_status()

    def _on_text_focus_in(self, _):
        if self.text_input.get("1.0", "end-1c") == self._placeholder:
            self.text_input.delete("1.0", "end")
            self.text_input.config(fg=TEXT_DARK)

    def _on_text_focus_out(self, _):
        if not self.text_input.get("1.0", "end-1c").strip():
            self.text_input.insert("1.0", self._placeholder)
            self.text_input.config(fg=TEXT_DIM)

    def _update_w2v_status(self):
        if self._w2v_model is not None:
            self._w2v_status.config(text="Word2Vec ✔ loaded", fg=SUCCESS)
        elif os.path.exists(W2V_MODEL):
            self._w2v_status.config(text="Word2Vec model found — loading on first extract", fg=WARNING)
        else:
            self._w2v_status.config(text="Word2Vec model not found (run pipeline first)", fg=ERROR)

    def _clear_kw_cols(self):
        for body in self._kw_cols.values():
            for w in body.winfo_children():
                w.destroy()

    def _clear_extractor(self):
        self._clear_kw_cols()
        self.text_input.delete("1.0", "end")
        self.text_input.insert("1.0", self._placeholder)
        self.text_input.config(fg=TEXT_DIM)

    def _extract_keywords(self):
        raw = self.text_input.get("1.0", "end-1c").strip()
        if not raw or raw == self._placeholder:
            messagebox.showwarning("No Text", "Please enter some text first.")
            return

        self._clear_kw_cols()
        top_n  = self.top_n_var.get()
        tokens = preprocess_text(raw)

        if len(tokens) < 3:
            messagebox.showwarning("Too Short",
                "Text too short after preprocessing. Try a longer passage.")
            return

        # TF-IDF style
        tfidf_kws = tfidf_style_keywords(tokens, top_n)
        self._render_keywords("tfidf", tfidf_kws, ACCENT)

        # Word2Vec
        if MODELS_AVAILABLE and os.path.exists(W2V_MODEL):
            if self._w2v_model is None:
                self.status_var.set("Loading Word2Vec model…")
                try:
                    self._w2v_model = Word2Vec.load(W2V_MODEL)
                    self.status_var.set("Word2Vec loaded ✔")
                    self._update_w2v_status()
                except Exception as e:
                    self.status_var.set(f"W2V load error: {e}")
            w2v_kws = w2v_style_keywords(tokens, self._w2v_model, top_n)
        else:
            w2v_kws = []

        if w2v_kws:
            self._render_keywords("w2v", w2v_kws, ACCENT2)
        else:
            body = self._kw_cols["w2v"]
            tk.Label(body,
                     text="Run the pipeline first to\ntrain the Word2Vec model.",
                     font=FONT_BODY, bg=BG_CARD, fg=TEXT_DIM,
                     justify="center").pack(pady=20)

        self.status_var.set(f"Extracted top-{top_n} keywords from {len(tokens)} tokens")

    def _render_keywords(self, key: str, kw_list: list, accent_color: str):
        body = self._kw_cols[key]
        for rank, (kw, score) in enumerate(kw_list, start=1):
            row = tk.Frame(body, bg=BG_CARD, pady=3)
            row.pack(fill="x")

            # Rank badge
            badge = tk.Label(row,
                             text=f"{rank:2d}",
                             font=("Segoe UI", 9, "bold"),
                             bg=accent_color, fg="white",
                             width=3, padx=4, pady=2)
            badge.pack(side="left")

            # Keyword text
            tk.Label(row, text=f"  {kw}",
                     font=FONT_KW, bg=BG_CARD, fg=TEXT_DARK,
                     anchor="w").pack(side="left", fill="x", expand=True)

            # Score bar
            bar_bg = tk.Frame(row, bg="#e8ecfb", height=8, width=80)
            bar_bg.pack(side="right", padx=(0, 4))
            bar_bg.pack_propagate(False)
            fill_w = max(4, int(score * 80))
            tk.Frame(bar_bg, bg=accent_color, width=fill_w, height=8
                     ).place(x=0, y=0, relheight=1)

            tk.Label(row, text=f"{score:.3f}",
                     font=FONT_SMALL, bg=BG_CARD, fg=TEXT_DIM,
                     width=5).pack(side="right")

            tk.Frame(body, bg=BORDER, height=1).pack(fill="x")


    # ─────────────────────────────────────────────────────────────────
    # SECTION 3 — RESULTS VIEWER
    # ─────────────────────────────────────────────────────────────────
    def _build_results_section(self):
        parent = self.sections["results"]

        # Header
        hdr = tk.Frame(parent, bg=BG_MAIN, pady=20, padx=30)
        hdr.pack(fill="x")
        tk.Label(hdr, text="📊  Results Viewer", font=FONT_TITLE,
                 bg=BG_MAIN, fg=TEXT_DARK).pack(side="left")

        # Tab bar for Charts / Summary Table
        self._results_tab = tk.StringVar(value="charts")
        tab_row = tk.Frame(hdr, bg=BG_MAIN)
        tab_row.pack(side="right")
        for key, label in [("charts", "Charts"), ("table", "Summary Table")]:
            b = make_btn(tab_row, label,
                         command=lambda k=key: self._switch_results_tab(k),
                         bg=ACCENT if key=="charts" else "#e8ecfb",
                         fg=TEXT_LIGHT if key=="charts" else ACCENT,
                         font=("Segoe UI", 9, "bold"),
                         padx=14, pady=6)
            b.pack(side="left", padx=2)
        self._results_tab_btns: dict[str, tk.Button] = {}

        # ── Charts panel ─────────────────────────────────────────────
        self._charts_panel = tk.Frame(parent, bg=BG_MAIN)
        self._charts_panel.pack(fill="both", expand=True, padx=30)

        # Navigation bar
        nav = tk.Frame(self._charts_panel, bg=BG_MAIN, pady=8)
        nav.pack(fill="x")

        make_btn(nav, "◀ Prev", self._prev_fig,
                 bg="#e8ecfb", fg=ACCENT, padx=16, pady=6).pack(side="left")
        make_btn(nav, "Next ▶", self._next_fig,
                 bg="#e8ecfb", fg=ACCENT, padx=16, pady=6).pack(side="left", padx=8)

        self._fig_title_var = tk.StringVar(value="")
        tk.Label(nav, textvariable=self._fig_title_var,
                 font=FONT_HEAD, bg=BG_MAIN, fg=TEXT_DARK).pack(side="left", padx=16)

        self._fig_counter_var = tk.StringVar(value="")
        tk.Label(nav, textvariable=self._fig_counter_var,
                 font=FONT_SMALL, bg=BG_MAIN, fg=TEXT_DIM).pack(side="right")

        # Image display area
        img_card = make_card(self._charts_panel)
        img_card.pack(fill="both", expand=True, pady=6)
        self._img_label = tk.Label(img_card, bg=BG_CARD,
                                   text="Run the pipeline first to generate charts.",
                                   font=FONT_BODY, fg=TEXT_DIM)
        self._img_label.pack(expand=True)

        # ── Summary table panel ───────────────────────────────────────
        self._table_panel = tk.Frame(parent, bg=BG_MAIN)
        self._table_label = tk.Label(
            self._table_panel,
            text="Run the pipeline (Step 5) to generate evaluation_results.csv",
            font=FONT_BODY, fg=TEXT_DIM, bg=BG_MAIN)
        self._table_label.pack(pady=20)

        # Build the Treeview for the CSV table
        cols = ["Doc","Sentiment","TF-IDF F1","Cosine F1","Cluster F1"]
        style = ttk.Style()
        style.configure("Custom.Treeview",
                         background=BG_CARD, foreground=TEXT_DARK,
                         rowheight=28, fieldbackground=BG_CARD,
                         font=FONT_BODY)
        style.configure("Custom.Treeview.Heading",
                         background=ACCENT, foreground="white",
                         font=("Segoe UI", 10, "bold"))
        style.map("Custom.Treeview", background=[("selected", "#d0d9ff")])

        tv_frame = tk.Frame(self._table_panel, bg=BG_MAIN, padx=30)
        tv_frame.pack(fill="both", expand=True)

        self._tv = ttk.Treeview(tv_frame, columns=cols, show="headings",
                                 style="Custom.Treeview")
        col_widths = {"Doc":40, "Sentiment":80,
                      "TF-IDF F1":100, "Cosine F1":100, "Cluster F1":100}
        for c in cols:
            self._tv.heading(c, text=c)
            self._tv.column(c, width=col_widths.get(c, 120), anchor="center")

        vsb = ttk.Scrollbar(tv_frame, orient="vertical",   command=self._tv.yview)
        self._tv.configure(yscrollcommand=vsb.set)
        self._tv.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._switch_results_tab("charts")

    def _switch_results_tab(self, key: str):
        self._results_tab.set(key)
        if key == "charts":
            self._table_panel.pack_forget()
            self._charts_panel.pack(fill="both", expand=True, padx=30)
        else:
            self._charts_panel.pack_forget()
            self._table_panel.pack(fill="both", expand=True)
            self._load_summary_table()

    def _fig_keys(self):
        return list(EVAL_FIGS.keys())

    def _prev_fig(self):
        keys = self._fig_keys()
        self._current_fig_idx = (self._current_fig_idx - 1) % len(keys)
        self._load_current_fig()

    def _next_fig(self):
        keys = self._fig_keys()
        self._current_fig_idx = (self._current_fig_idx + 1) % len(keys)
        self._load_current_fig()

    def _load_current_fig(self):
        keys     = self._fig_keys()
        key      = keys[self._current_fig_idx]
        path     = EVAL_FIGS[key]
        label    = FIG_LABELS.get(key, key)
        total    = len(keys)
        idx      = self._current_fig_idx + 1

        self._fig_title_var.set(f"{label}")
        self._fig_counter_var.set(f"Figure {idx} / {total}")

        if not os.path.exists(path):
            self._img_label.config(image="", text=f"Chart not found:\n{path}\n\nRun the full pipeline first.",
                                   compound="none")
            return

        if not PIL_AVAILABLE:
            self._img_label.config(image="", text=f"Install Pillow to view charts:\n  pip install pillow",
                                   compound="none")
            return

        try:
            # Fit image to available label space
            img  = Image.open(path)
            W, H = self._img_label.winfo_width() or 900, self._img_label.winfo_height() or 480
            W, H = max(W, 600), max(H, 400)
            img.thumbnail((W - 20, H - 20), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._img_label.config(image=photo, text="", compound="none")
            self._img_label._photo = photo          # prevent GC
        except Exception as e:
            self._img_label.config(image="", text=f"Could not load image: {e}", compound="none")

    def _load_summary_table(self):
        for row in self._tv.get_children():
            self._tv.delete(row)

        if not os.path.exists(EVAL_RESULTS_CSV):
            self._table_label.config(
                text="Run the pipeline (Step 5) to generate evaluation_results.csv")
            return

        try:
            import csv
            self._table_label.config(text="")
            with open(EVAL_RESULTS_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    vals = (
                        row.get("Doc",""),
                        row.get("Sentiment",""),
                        row.get("TF-IDF F1",""),
                        row.get("Cosine F1",""),
                        row.get("Cluster F1",""),
                    )
                    tag = "even" if i % 2 == 0 else "odd"
                    self._tv.insert("", "end", values=vals, tags=(tag,))
            self._tv.tag_configure("even", background="#f0f4ff")
            self._tv.tag_configure("odd",  background=BG_CARD)
        except Exception as e:
            self._table_label.config(text=f"Error loading CSV: {e}")


    # ─────────────────────────────────────────────────────────────────
    # SECTION 4 — ABOUT
    # ─────────────────────────────────────────────────────────────────
    def _build_about_section(self):
        parent = self.sections["about"]

        tk.Frame(parent, bg=BG_MAIN, height=40).pack()

        card = make_card(parent, padx=40, pady=36)
        card.pack(padx=80, pady=20, fill="x")

        tk.Label(card, text="🔑 Keyword Extraction System", font=FONT_TITLE,
                 bg=BG_CARD, fg=TEXT_DARK).pack(anchor="w")
        tk.Label(card, text="NLP Project 5  |  FCAI", font=FONT_HEAD,
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", pady=(4, 20))

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=(0, 20))

        info_rows = [
            ("Dataset",   "IMDB Movie Reviews  (50 000 documents)"),
            ("Methods",   "TF-IDF Baseline  ·  Cosine Re-ranking  ·  KMeans Clustering"),
            ("Embeddings","Word2Vec (vector_size=100, window=5, epochs=10)"),
            ("Evaluation","Precision / Recall / F1 vs 25 manually annotated documents"),
            ("Metrics",   "Confusion Matrix  ·  TP / FP / FN  ·  9 evaluation charts"),
        ]
        for label, value in info_rows:
            row = tk.Frame(card, bg=BG_CARD, pady=4)
            row.pack(fill="x")
            tk.Label(row, text=f"{label}:", font=("Segoe UI", 10, "bold"),
                     bg=BG_CARD, fg=TEXT_DARK, width=14, anchor="w").pack(side="left")
            tk.Label(row, text=value, font=FONT_BODY,
                     bg=BG_CARD, fg=TEXT_DIM, anchor="w").pack(side="left")

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=20)

        # File status grid
        tk.Label(card, text="File Status", font=FONT_HEAD,
                 bg=BG_CARD, fg=TEXT_DARK).pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(card, bg=BG_CARD)
        grid.pack(fill="x")
        all_files = [
            ("Dataset",       DATASET),
            *[(os.path.basename(f), f) for f in ALL_INTERMEDIATE],
        ]
        for i, (name, path) in enumerate(all_files):
            exists = os.path.exists(path)
            col = i % 2
            row_i = i // 2
            cell = tk.Frame(grid, bg="#f0fff4" if exists else "#fff0f0",
                            padx=8, pady=4,
                            highlightthickness=1,
                            highlightbackground="#c3e6cb" if exists else "#f5c6cb")
            cell.grid(row=row_i, column=col, padx=4, pady=2, sticky="ew")
            grid.columnconfigure(col, weight=1)
            icon = "✔" if exists else "✘"
            clr  = SUCCESS if exists else ERROR
            tk.Label(cell, text=icon, fg=clr, bg=cell["bg"],
                     font=("Segoe UI", 11, "bold")).pack(side="left")
            tk.Label(cell, text=f"  {name}", font=FONT_SMALL,
                     bg=cell["bg"], fg=TEXT_DARK).pack(side="left")

        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", pady=20)
        make_btn(card, "🔄  Refresh Status",
                 self._refresh_about,
                 bg="#e8ecfb", fg=ACCENT,
                 font=FONT_BODY, padx=16, pady=6).pack(anchor="w")

    def _refresh_about(self):
        # Re-build the about section to refresh file status
        f = self.sections["about"]
        for w in f.winfo_children():
            w.destroy()
        self._build_about_section()


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()