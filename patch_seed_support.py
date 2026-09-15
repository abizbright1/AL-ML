#!/usr/bin/env python3
"""
patch_seed_support.py — one-shot patch of run_full_experiment.py
=================================================================

Applies three changes required for a multi-seed study:

  1. set_all_seeds()  — seeds python `random`, numpy, torch, torch.cuda,
     sets cudnn determinism flags, and returns a seeded DataLoader generator.
     The current code seeds only torch and numpy (lines 960-961) and leaves
     both DataLoaders unseeded.

  2. --models  — comma-separated filter on the `short` label in
     build_models(), so a sweep can train only the four paper configurations
     instead of all nine.

  3. Seed-stamped output directory — if --out does not already contain
     "seed", "_seed{N}" is appended, so sweep runs cannot overwrite one
     another.

Does NOT touch the loss, the model, or the split logic.
Idempotent: re-running is a no-op (asserts fail loudly if already applied).

Usage
-----
    python3 patch_seed_support.py
    python3 -c "import ast; ast.parse(open('run_full_experiment.py').read())"
"""

import sys

P = 'run_full_experiment.py'
s = open(P).read()

if 'def set_all_seeds' in s:
    sys.exit('Already patched — set_all_seeds() is present. Nothing to do.')

# ---------------------------------------------------------------- 1. imports
old = """import argparse
import csv
import json
import os
import subprocess
import sys
import time"""
new = """import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time"""
assert s.count(old) == 1, 'import anchor not found'
s = s.replace(old, new)

# ------------------------------------------------------------------ 2. --models
old = """    p.add_argument('--figures_only', action='store_true',
                   help='Skip training; redraw figures from saved .npz dumps')
    return p.parse_args()"""
new = """    p.add_argument('--figures_only', action='store_true',
                   help='Skip training; redraw figures from saved .npz dumps')
    p.add_argument('--models', type=str, default=None,
                   help="Comma-separated 'short' labels to train, e.g. "
                        "'SwinIR-L1,Uformer-L1,SwinIR+PL,Uformer+PL'. "
                        "Default: every model in build_models().")
    return p.parse_args()"""
assert s.count(old) == 1, '--models anchor not found'
s = s.replace(old, new)

# ------------------------------------------------------- 3. seeding utilities
old = """# ═══════════════════════════════════════════════════════════════════════════
#  Provenance — so you can always tell two runs apart
# ═══════════════════════════════════════════════════════════════════════════"""
new = '''# ═══════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int) -> torch.Generator:
    """Seed every RNG this run touches; return a seeded DataLoader generator.

    The previous code seeded only torch and numpy. Three further sources of
    randomness were left free:
      * python's stdlib `random` (used by the subject-split shuffle)
      * torch.cuda (irrelevant on MPS, included for portability)
      * each DataLoader's own generator, which drives shuffle order

    NOTE ON MPS: Apple's Metal backend provides no determinism guarantee and
    has no equivalent of cudnn.deterministic. Two runs with the same seed on
    MPS can still differ slightly. The seed makes runs comparable, not
    bit-identical. This is stated in the run banner.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible RNG stream.

    Inert while num_workers=0 (the current default); present so that raising
    num_workers later does not silently reintroduce nondeterminism.
    """
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def seed_stamped(out_dir: str, seed: int) -> str:
    """Append _seed{N} unless the path already identifies a seed."""
    return out_dir if 'seed' in os.path.basename(out_dir).lower() \\
        else f'{out_dir}_seed{seed}'


# ═══════════════════════════════════════════════════════════════════════════
#  Provenance — so you can always tell two runs apart
# ═══════════════════════════════════════════════════════════════════════════'''
assert s.count(old) == 1, 'seeding-utilities anchor not found'
s = s.replace(old, new)

# ------------------------------------------- 4. call set_all_seeds in main()
old = """    torch.manual_seed(args.seed)
    np.random.seed(args.seed)"""
new = """    _loader_gen = set_all_seeds(args.seed)"""
assert s.count(old) == 1, 'seed-call anchor not found'
s = s.replace(old, new)

# ------------------------------------------------- 5. seed-stamp the out dir
old = """def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)"""
new = """def main():
    args = parse_args()
    if not args.figures_only:
        args.out = seed_stamped(args.out, args.seed)
    os.makedirs(args.out, exist_ok=True)"""
assert s.count(old) == 1, 'out-dir anchor not found'
s = s.replace(old, new)

# ---------------------------------------------------- 6. seed the DataLoaders
old = """    train_loader = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   args.batch_size, shuffle=False)"""
new = """    train_loader = torch.utils.data.DataLoader(
        train_ds, args.batch_size, shuffle=True,
        generator=_loader_gen, worker_init_fn=_worker_init)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   args.batch_size, shuffle=False,
        worker_init_fn=_worker_init)"""
assert s.count(old) == 1, 'DataLoader anchor not found'
s = s.replace(old, new)

# ------------------------------------------------------- 7. apply --models
old = """    specs, results, histories, all_slices = build_models(device), [], {}, []"""
new = """    specs, results, histories, all_slices = build_models(device), [], {}, []

    if args.models:
        keep = {m.strip() for m in args.models.split(',') if m.strip()}
        avail = {sp['short'] for sp in specs.values()}
        missing = keep - avail
        if missing:
            sys.exit(f"--models: unknown label(s) {sorted(missing)}. "
                     f"Available: {sorted(avail)}")
        specs = {k: v for k, v in specs.items() if v['short'] in keep}
        print(f"  training {len(specs)} of {len(avail)} models: "
              f"{sorted(sp['short'] for sp in specs.values())}\\n", flush=True)"""
assert s.count(old) == 1, 'model-filter anchor not found'
s = s.replace(old, new)

# ------------------------------------------------ 8. determinism note in banner
old = """    print(f"Git SHA  : {git_sha()[:8]}{'  (DIRTY)' if git_dirty() else ''}")"""
new = """    print(f"Seed     : {args.seed}"
          f"{'   (MPS: seeded, not bit-deterministic)' if device == 'mps' else ''}")
    print(f"Git SHA  : {git_sha()[:8]}{'  (DIRTY)' if git_dirty() else ''}")"""
assert s.count(old) == 1, 'banner anchor not found'
s = s.replace(old, new)

open(P, 'w').write(s)
print('PATCHED run_full_experiment.py')
print('  + set_all_seeds() / _worker_init() / seed_stamped()')
print('  + --models filter')
print('  + seed-stamped output directory')
print('  + DataLoader generator and worker_init_fn')
