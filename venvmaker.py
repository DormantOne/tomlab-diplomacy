#!/usr/bin/env python3
"""
Centralized venv manager (numbered menu)
- Master directory (default):  ~/Desktop/VENV_MASTER
    Override with env var: VENV_MASTER_DIR=/custom/path
- Each environment lives as:   <MASTER>/<env-name>/venv
- Actions: List / Create / Activate / Deactivate (help) / Freeze / Delete / Open master

Activation opens an INTERACTIVE SUBSHELL with the venv active and shows "(envname)" in the prompt.
Type `deactivate` to drop the venv (stay in subshell) or `exit` to return here.
"""

import os, re, shutil, subprocess, sys
from pathlib import Path

# ----- Config -----
SPECIAL_VENV_DIRNAME = "venv"  # fixed inner name per environment
MASTER_DEFAULT = Path.home() / "Desktop" / "VENV_MASTER"
MASTER_DIR = Path(os.environ.get("VENV_MASTER_DIR", str(MASTER_DEFAULT))).expanduser()

MASTER_DIR.mkdir(parents=True, exist_ok=True)

def shell_path_and_name():
    sh = os.environ.get("SHELL", "/bin/zsh")
    return sh, os.path.basename(sh)

def sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name or "env"

def env_root(name: str) -> Path:
    """Parent directory for an env (contains the fixed inner 'venv' dir)."""
    return MASTER_DIR / name

def env_dir(name: str) -> Path:
    """Actual Python venv directory path: <MASTER>/<name>/venv"""
    return env_root(name) / SPECIAL_VENV_DIRNAME

def _activation_candidates(name: str):
    """
    Return dict of possible activation scripts for bash/zsh, fish, csh (new & legacy layout).
    New layout: <root>/<name>/venv/bin/activate*
    Legacy (fallback): <root>/<name>/bin/activate*
    """
    new_base = env_dir(name) / "bin"
    old_base = env_root(name) / "bin"  # legacy
    return {
        "posix": [new_base / "activate", old_base / "activate"],
        "fish":  [new_base / "activate.fish", old_base / "activate.fish"],
        "csh":   [new_base / "activate.csh", old_base / "activate.csh"],
        "pip":   [new_base / "pip", old_base / "pip"],
    }

def list_envs():
    envs = []
    if not MASTER_DIR.exists():
        return envs
    for p in sorted(MASTER_DIR.iterdir()):
        if not p.is_dir():
            continue
        # consider it a valid env if new or legacy layout exists
        if (p / SPECIAL_VENV_DIRNAME / "bin" / "activate").exists() or (p / "bin" / "activate").exists():
            envs.append(p.name)
    return envs

def choose_env(prompt="Select env number"):
    envs = list_envs()
    if not envs:
        print("— no environments yet —")
        return None
    print(f"(master: {MASTER_DIR})")
    for i, name in enumerate(envs, 1):
        print(f"  {i}) {name}")
    sel = input(f"{prompt}: ").strip()
    try:
        ix = int(sel)
        if 1 <= ix <= len(envs):
            return envs[ix - 1]
    except Exception:
        pass
    print("Invalid selection.")
    return None

def create_env():
    py = shutil.which("python3") or shutil.which("python")
    if not py:
        print("No python interpreter found on PATH.")
        return
    name = sanitize(input("New env name (e.g., tiny, llm, tools): "))
    root = env_root(name)
    vdir = env_dir(name)
    if (vdir / "bin" / "activate").exists():
        print(f"Env '{name}' already exists at {vdir}")
        return
    root.mkdir(parents=True, exist_ok=True)
    print(f"Creating venv '{name}' at {vdir} …")
    try:
        subprocess.check_call([py, "-m", "venv", str(vdir)])
        print(f"✅ Created: {vdir}")
        print("Tip: choose 'Activate environment' to enter it.")
    except subprocess.CalledProcessError as e:
        print("Failed to create venv:", e)

def _posix_activation_cmd(act_path: Path, env_name: str, shell_path: str) -> str:
    # For bash/zsh: source, ensure prompt shows (env), then start interactive subshell
    return f'''
export VIRTUAL_ENV_DISABLE_PROMPT=0
source "{act_path}"
# Prepend venv name to prompt if theme doesn't show it
if [ -n "$ZSH_VERSION" ]; then
  export PROMPT="({env_name}) $PROMPT"
elif [ -n "$BASH_VERSION" ]; then
  export PS1="({env_name}) $PS1"
fi
echo "✅ Activated {env_name}. Type: deactivate  or  exit"
"{shell_path}" -i
'''

def _fish_activation_cmd(act_path: Path, env_name: str, shell_path: str) -> str:
    # For fish: source, wrap fish_prompt to prepend (env), then start interactive subshell
    return f'''
source "{act_path}"
set -gx VIRTUAL_ENV_DISABLE_PROMPT 0
functions -q __orig_fish_prompt; or functions -c fish_prompt __orig_fish_prompt
function fish_prompt
  echo -n "({env_name}) "
  __orig_fish_prompt
end
echo "✅ Activated {env_name} (fish). Type: deactivate  or  exit"
"{shell_path}" -i
'''

def _csh_activation_cmd(act_path: Path, env_name: str, shell_path: str) -> str:
    # For csh/tcsh: source, prefix prompt, start interactive subshell
    return f'''
source "{act_path}"
setenv VIRTUAL_ENV_DISABLE_PROMPT 0
set prompt="({env_name}) $prompt"
echo "✅ Activated {env_name} (csh). Type: deactivate  or  exit"
"{shell_path}" -i
'''

def _find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def activate_env():
    name = choose_env("Activate which env")
    if not name:
        return
    sh, sh_base = shell_path_and_name()
    cands = _activation_candidates(name)

    if sh_base == "fish":
        act = _find_first_existing(cands["fish"])
        if not act:
            print("Could not find fish activate script.")
            return
        cmd = _fish_activation_cmd(act, name, sh)
    elif sh_base in ("csh", "tcsh"):
        act = _find_first_existing(cands["csh"])
        if not act:
            print("Could not find csh/tcsh activate script.")
            return
        cmd = _csh_activation_cmd(act, name, sh)
    else:
        act = _find_first_existing(cands["posix"])
        if not act:
            print("Could not find POSIX activate script.")
            return
        cmd = _posix_activation_cmd(act, name, sh)

    print(f"\n— entering subshell for env: {name} —")
    print("When done: type `exit` to return to this menu.\n")
    try:
        subprocess.call([sh, "-c", cmd])
    except KeyboardInterrupt:
        pass
    print(f"\n— left env: {name} —\n")

def show_deactivate_help():
    print("""
Deactivation notes
------------------
• If you're inside the activated subshell:
    - Type:  deactivate    # drops the venv but keeps the subshell
      (or)
    - Type:  exit          # closes subshell and returns to this menu

• If you're back at this menu already, you're NOT in a venv anymore.
""".strip())

def freeze_requirements():
    name = choose_env("Freeze requirements for which env")
    if not name:
        return
    cands = _activation_candidates(name)
    pip = _find_first_existing(cands["pip"])
    if not pip:
        print("pip not found in env.")
        return

    default_out = Path.cwd() / f"requirements-{name}.txt"
    out = input(f"Output file [{default_out.name}]: ").strip() or default_out.name
    out_path = Path.cwd() / out
    print(f"Freezing packages from '{name}' → {out_path}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            subprocess.check_call([str(pip), "freeze"], stdout=f)
        print(f"✅ Wrote {out_path}")
    except subprocess.CalledProcessError as e:
        print("pip freeze failed:", e)

def delete_env():
    name = choose_env("Delete which env")
    if not name:
        return
    root = env_root(name)
    confirm = input(f"Type '{name}' to confirm delete: ").strip()
    if confirm != name:
        print("Canceled.")
        return
    shutil.rmtree(root, ignore_errors=True)
    print(f"🗑️  Deleted: {name}")

def open_master():
    if sys.platform == "darwin":
        subprocess.call(["open", str(MASTER_DIR)])
    else:
        print(f"Master folder: {MASTER_DIR}")

def list_envs_action():
    envs = list_envs()
    print(f"Master folder: {MASTER_DIR}")
    if envs:
        print("Environments:")
        for e in envs:
            print(f"  - {e}")
    else:
        print("— none yet —")

def menu():
    while True:
        print("=== centralized venv manager ===")
        print(f"(master: {MASTER_DIR})")
        print(" 1) List environments")
        print(" 2) Create new environment")
        print(" 3) Activate environment")
        print(" 4) Deactivate (how-to)")
        print(" 5) Freeze requirements (pip freeze)")
        print(" 6) Delete environment")
        print(" 7) Open master folder")
        print(" 0) Exit")
        choice = input("Select: ").strip()
        if choice == "1": list_envs_action()
        elif choice == "2": create_env()
        elif choice == "3": activate_env()
        elif choice == "4": show_deactivate_help()
        elif choice == "5": freeze_requirements()
        elif choice == "6": delete_env()
        elif choice == "7": open_master()
        elif choice == "0": break
        else:
            print("Unknown selection.")
        print()

if __name__ == "__main__":
    menu()
