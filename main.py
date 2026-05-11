"""
╔══════════════════════════════════════════════════════════════════════╗
║          PROJECT 5 — KEYWORD EXTRACTION SYSTEM                      ║
║          MAIN PIPELINE RUNNER                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Runs all 5 steps in order:                                          ║
║    Step 1 -> preprocessing.py                                        ║
║    Step 2 -> TF_IDF.py                                              ║
║    Step 3 -> word_embeddings.py                                      ║
║    Step 4 -> advanced_keyword_extraction.py                          ║
║    Step 5 -> evaluation.py                                           ║
║                                                                      ║
║  HOW TO RUN:                                                         ║
║    1. Place IMDB-Dataset.csv next to this script.                    ║
║    2. pip install pandas numpy scikit-learn gensim                   ║
║                   matplotlib seaborn nltk tqdm                       ║
║    3. python main.py                                                 ║
║                                                                      ║
║  Folder structure after running:                                     ║
║    generated/           <- all intermediate files                    ║
║    evaluation_results/  <- charts + final CSV                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import subprocess
from paths import DATASET, GEN_DIR, EVAL_DIR, ALL_INTERMEDIATE, ALL_EVAL_OUTPUTS

PIPELINE = [
    ("Step 1 — Enhanced Preprocessing",           "preprocessing.py"),
    ("Step 2 — TF-IDF Feature Extraction",       "TF_IDF.py"),
    ("Step 3 — Word2Vec Embeddings",             "word_embeddings.py"),
    ("Step 4 — Advanced Keyword Extraction",     "advanced_keyword_extraction.py"),
    ("Step 5 — Evaluation & Comparison",         "evaluation.py"),
]

# Simplified enhanced pipeline (just KeyBERT for state-of-the-art comparison)
ENHANCED_PIPELINE = [
    ("Step 6 — KeyBERT Extraction",             "keybert_model.py"),
]

# ══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECK
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  KEYWORD EXTRACTION SYSTEM — Full Pipeline")
print("=" * 65)
print(f"\n  Intermediate files -> {GEN_DIR}/")
print(f"  Evaluation outputs -> {EVAL_DIR}/")

if not os.path.exists(DATASET):
    print(f"\n  [ERROR] Dataset not found: '{DATASET}'")
    print("  -> Download the IMDB Dataset from:")
    print("     https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print("  -> Place IMDB-Dataset.csv in the same folder as main.py")
    sys.exit(1)

print(f"\n  Dataset found : {DATASET}")
print(f"  Python        : {sys.executable}")
print(f"  Basic steps   : {len(PIPELINE)}")
print(f"  Enhanced steps: {len(ENHANCED_PIPELINE)}")

# Check for enhanced mode
enhanced_mode = '--enhanced' in sys.argv or '-e' in sys.argv
if enhanced_mode:
    print(f"\n  🚀 RUNNING IN ENHANCED MODE")
    print(f"  All optimization and advanced features enabled")
else:
    print(f"\n  📊 RUNNING IN BASIC MODE")
    print(f"  Use --enhanced or -e for full features")
print()

# ══════════════════════════════════════════════════════════════════════
# RUN EACH STEP
# ══════════════════════════════════════════════════════════════════════
step_times    = []
pipeline_start = time.time()

for step_num, (step_name, script) in enumerate(PIPELINE, start=1):
    print("=" * 65)
    print(f"  [{step_num}/{len(PIPELINE)}]  {step_name}")
    print(f"  Running: python {script}")
    print("=" * 65)

    if not os.path.exists(script):
        print(f"\n  [ERROR] Script not found: '{script}'")
        sys.exit(1)

    t0     = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    step_times.append(elapsed)

    if result.returncode != 0:
        print(f"\n  [FAILED] {script} exited with code {result.returncode}")
        print("  -> Fix the error above and re-run main.py")
        sys.exit(result.returncode)

    print(f"\n  Step completed in {elapsed:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
total = time.time() - pipeline_start

print("=" * 65)
print("  PIPELINE COMPLETE — All steps finished successfully!")
print("=" * 65)

print("\n  Step timings:")
for (name,_),t in zip(PIPELINE,step_times):
    print(f"    {name:<44} {t:6.1f}s")
print(f"    {'TOTAL':<44} {total:6.1f}s")

print(f"\n  [{GEN_DIR}/]  — Intermediate files:")
for f in ALL_INTERMEDIATE:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"    [{status:^7}]  {os.path.basename(f)}")

print(f"\n  [{EVAL_DIR}/]  — Evaluation outputs:")
for f in ALL_EVAL_OUTPUTS:
    status = "OK" if os.path.exists(f) else "MISSING"
    print(f"    [{status:^7}]  {os.path.basename(f)}")

# ══════════════════════════════════════════════════════════════════════
# ENHANCED PIPELINE (if enabled)
# ══════════════════════════════════════════════════════════════════════
if enhanced_mode:
    print("\n" + "=" * 65)
    print("  ENHANCED PIPELINE — Advanced Features & Optimization")
    print("=" * 65)
    
    enhanced_step_times = []
    enhanced_start = time.time()
    
    for step_num, (step_name, script) in enumerate(ENHANCED_PIPELINE, start=len(PIPELINE)+1):
        print("=" * 65)
        print(f"  [{step_num}/{len(PIPELINE)+len(ENHANCED_PIPELINE)}]  {step_name}")
        print(f"  Running: python {script}")
        print("=" * 65)
        
        if not os.path.exists(script):
            print(f"\n  [WARNING] Script not found: '{script}' - Skipping")
            continue
        
        t0 = time.time()
        try:
            result = subprocess.run([sys.executable, script], 
                                  capture_output=True, text=True, timeout=300)
            elapsed = time.time() - t0
            enhanced_step_times.append(elapsed)
            
            if result.returncode != 0:
                print(f"\n  [WARNING] {script} failed with code {result.returncode}")
                print(f"  Error: {result.stderr}")
            else:
                print(f"\n  Step completed in {elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"\n  [TIMEOUT] {script} timed out after 5 minutes")
        except Exception as e:
            print(f"\n  [ERROR] Failed to run {script}: {e}")
    
    enhanced_total = time.time() - enhanced_start
    
    print("\n" + "=" * 65)
    print("  ENHANCED PIPELINE SUMMARY")
    print("=" * 65)
    
    print("\n  Enhanced step timings:")
    for (name, _), t in zip(ENHANCED_PIPELINE, enhanced_step_times):
        print(f"    {name:<44} {t:6.1f}s")
    print(f"    {'ENHANCED TOTAL':<44} {enhanced_total:6.1f}s")

print()
print("=" * 65)