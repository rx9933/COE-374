#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing Homebrew (if not installed)..."
if ! command -v brew >/dev/null 2>&1; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Updating Homebrew..."
brew update

echo "Installing core dependencies and pyenv..."
brew install \
  git \
  curl \
  wget \
  cmake \
  pkg-config \
  openssl \
  readline \
  sqlite3 \
  xz \
  zlib \
  tcl-tk \
  libffi \
  llvm \
  opencv \
  docker \
  pyenv

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "Setting up pyenv..."
if ! grep -q 'pyenv init' ~/.zshrc 2>/dev/null; then
  cat <<'EOF' >> ~/.zshrc

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
fi

echo "Installing nvm..."
if [ ! -d "$HOME/.nvm" ]; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
fi

export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1090
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

echo "Installing rustup..."
if ! command -v rustup >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  export PATH="$HOME/.cargo/bin:$PATH"
fi

PYTHON_FILE="$SCRIPT_DIR/.python-version"
if [ -f "$PYTHON_FILE" ]; then
  PYTHON_VERSION="$(cat "$PYTHON_FILE" | tr -d '[:space:]')"
  echo "Installing Python $PYTHON_VERSION..."
  pyenv install -s "$PYTHON_VERSION"
  pyenv local "$PYTHON_VERSION"
else
  echo "No .python-version file found."
fi

NODE_FILE="$SCRIPT_DIR/.nvmrc"
if [ -f "$NODE_FILE" ]; then
  NODE_VERSION="$(cat "$NODE_FILE" | tr -d '[:space:]')"
  echo "Installing Node $NODE_VERSION..."
  nvm install "$NODE_VERSION"
  nvm use "$NODE_VERSION"
else
  echo "No .nvmrc file found."
fi

if [ -d "$SCRIPT_DIR/website" ]; then
  echo "Installing Node dependencies in 'website'..."
  cd "$SCRIPT_DIR/website"
  npm ci
else
  echo "Website directory not found, skipping npm install."
fi

echo "Setup complete."
