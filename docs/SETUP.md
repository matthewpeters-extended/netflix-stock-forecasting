# Machine Setup — macOS

Everything that needs to be installed before Step 7 of [`WALKTHROUGH.md`](../WALKTHROUGH.md),
in order. Written for this machine specifically.

**Target machine (checked 2026-09-08):** macOS 26.6.2, Apple Silicon (`arm64`), zsh.
**Total time:** ~30–40 minutes, most of it downloading.

## What is already installed

| Tool | Status |
|---|---|
| Xcode Command Line Tools | Installed (26.6.0) — nothing to do |
| `git` | Installed (via Command Line Tools) |
| Python | **3.9.6 only**, the Apple system Python at `/usr/bin/python3` |
| pip | 21.2.4 (old, bundled with system Python) |
| Homebrew | Not installed |
| `gh` (GitHub CLI) | Not installed |
| pandas / numpy / Jupyter | Not installed |
| git identity (`user.name`, `user.email`) | **Not configured** — commits will fail until it is |

Two things to know about the system Python at `/usr/bin/python3`:

1. Apple ships it for macOS's own use. Installing packages into it can break system tooling, and
   Apple can replace it during an OS update without warning.
2. Version 3.9 was released in 2020 and sits at the very edge of what current TensorFlow supports.

So the plan is: leave system Python alone, install a modern Python alongside it, and put this
project in its own virtual environment.

---

## Step 1 — Xcode Command Line Tools

Already installed on this machine. Verify:

```bash
xcode-select -p
```

Expected: `/Library/Developer/CommandLineTools`

If it ever comes back empty on another machine, run `xcode-select --install` and accept the GUI
prompt. Homebrew needs this, so it has to come first.

---

## Step 2 — Homebrew

Homebrew is the package manager for macOS. It is how you install Python, `gh`, and essentially
every other developer tool without hunting for installers.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**This will ask for your Mac login password.** That is expected — it needs administrator rights to
create `/opt/homebrew`. It is the only step in this guide that requires a password.

On Apple Silicon, Homebrew installs to `/opt/homebrew`, which is **not** on the default `PATH`.
The installer prints instructions at the end; this is what they amount to:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

You do not currently have a `~/.zprofile` — the first command creates it. Terminal.app starts
login shells, so `~/.zprofile` is read on every new window. Your existing `~/.zshrc` is untouched.

Verify:

```bash
brew --version
```

Expected: a version number. If you get `command not found`, open a **new** terminal window and try
again — the `PATH` change only applies to shells started after it.

---

## Step 3 — Python 3.12

```bash
brew install python@3.12
```

**Why 3.12 and not the newest release.** Machine learning packages — TensorFlow above all — lag
behind new Python versions by six to twelve months, because they ship compiled binaries that have
to be rebuilt per version. On the newest Python you routinely hit "no matching distribution found"
and end up compiling from source or downgrading anyway. 3.12 is old enough that every package in
`requirements.txt` has a prebuilt wheel and new enough to be well ahead of the system's 3.9.

Verify:

```bash
/opt/homebrew/bin/python3.12 --version
```

Expected: `Python 3.12.x`

Leave `/usr/bin/python3` exactly as it is. Do not try to replace or override the system Python;
the virtual environment in Step 6 makes that unnecessary.

---

## Step 4 — GitHub CLI, and authenticating

```bash
brew install gh
gh auth login
```

`gh auth login` is interactive. Answer:

- **What account?** → GitHub.com
- **Preferred protocol?** → HTTPS
- **Authenticate Git with your GitHub credentials?** → **Yes** (this is the important one — it
  configures the credential helper so `git push` stops asking for a password)
- **How would you like to authenticate?** → Login with a web browser

It shows a one-time code, you press Enter, your browser opens, you paste the code and approve.

Verify:

```bash
gh auth status
```

Without this step, `git push` over HTTPS will fail. GitHub stopped accepting account passwords for
git operations in 2021 — you need either this, a personal access token, or an SSH key. `gh auth
login` is the least painful of the three.

---

## Step 5 — Git identity

Not configured on this machine yet, and git refuses to commit without it. These get stamped into
every commit you make, so use the name and email you want visible on your public repositories.

```bash
git config --global user.name "Matthew Peters"
git config --global user.email "matthewpeters711@gmail.com"
git config --global init.defaultBranch main
```

The email should match one registered on your GitHub account, otherwise commits will not be
attributed to your profile and will not show up on your contribution graph.

Verify:

```bash
git config --global --list
```

---

## Step 6 — The project virtual environment

A virtual environment is a private copy of Python and its packages, scoped to one project. It is
why installing TensorFlow here cannot break anything else on your machine, and it is why the
`requirements.txt` in this repository is enough for someone else to reproduce your setup.

```bash
cd ~/projects/netflix-stock-forecasting
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. While it is active, plain `python` and `pip` mean the
ones inside `.venv` — you no longer need the full path.

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

That last command pulls roughly 700 MB, mostly TensorFlow. Expect five to fifteen minutes.

`.venv/` is already in `.gitignore` — virtual environments are never committed. Anyone cloning the
repository rebuilds theirs from `requirements.txt`.

**Every time you come back to this project**, in a new terminal:

```bash
cd ~/projects/netflix-stock-forecasting && source .venv/bin/activate
```

Forgetting this is the single most common cause of "but I installed pandas already."

---

## Step 7 — Verify the install

```bash
python -c "import sys, pandas, numpy, sklearn, statsmodels, yfinance; print('python', sys.version.split()[0]); print('pandas', pandas.__version__)"
python -c "import tensorflow as tf; print('tensorflow', tf.__version__)"
```

The TensorFlow line is the one that matters — it is the only dependency with a real chance of
failing, and finding out now is much better than finding out in Step 12 of the walkthrough.

**If TensorFlow fails to install or import:**

- First try pinning it. Edit `requirements.txt` to `tensorflow>=2.16,<2.18` and re-run
  `pip install -r requirements.txt`.
- If it still fails, use PyTorch instead: `pip install torch`. The LSTM is about fifteen lines
  either way and nothing else in the project depends on Keras.
- Do not let this block you. Sessions 2, 3 and 4 — exploratory analysis, stationarity testing, and
  the baseline models — need none of it. Carry on and resolve it before Session 5.

Ignore any TensorFlow startup messages about `oneDNN` or CPU instruction sets. They are
informational, not errors.

---

## Step 8 — Jupyter

Already installed by `requirements.txt`. Register this environment as a named kernel so the
notebooks can find it:

```bash
python -m ipykernel install --user --name nflx --display-name "Python (nflx)"
jupyter lab
```

That opens JupyterLab in your browser. In any notebook, pick **Python (nflx)** as the kernel.
`Ctrl-C` twice in the terminal shuts it down.

---

## Step 9 — Connect the repository and push

```bash
cd ~/projects/netflix-stock-forecasting
git init
git add -A
git commit -m "Add plan, README, walkthrough, and setup guide"
git remote add origin https://github.com/matthewpeters-extended/netflix-stock-forecasting.git
git branch -M main
git push -u origin main
```

If GitHub created the repository with its own README, the push will be rejected because the
histories differ. Fix it with:

```bash
git pull --rebase origin main
git push -u origin main
```

Then confirm on <https://github.com/matthewpeters-extended/netflix-stock-forecasting> that
`README.md` renders on the front page.

---

## Step 10 — An editor (optional)

```bash
brew install --cask visual-studio-code
```

Then install the **Python** and **Jupyter** extensions from Microsoft inside VS Code. It will
detect `.venv` automatically and can run the notebooks without a browser. Skip this if you already
have an editor you like — nothing in the project requires it.

---

## Summary — the whole thing, in order

```bash
# 1. Homebrew (asks for your Mac password)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# 2. Python and the GitHub CLI
brew install python@3.12 gh

# 3. Authenticate (interactive) and set your git identity
gh auth login
git config --global user.name "Matthew Peters"
git config --global user.email "matthewpeters711@gmail.com"
git config --global init.defaultBranch main

# 4. Project environment
cd ~/projects/netflix-stock-forecasting
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Verify
python -c "import tensorflow as tf; print(tf.__version__)"

# 6. Publish
git init && git add -A && git commit -m "Add plan, README, walkthrough, and setup guide"
git remote add origin https://github.com/matthewpeters-extended/netflix-stock-forecasting.git
git branch -M main && git push -u origin main
```
