
init_venv () {
    name="$1"
    pythonver="$2"
    pipver="$3"
    echo "[$(date)] Initializing Python virtual environment '$name' using Python $pythonver and pip $pipver."
    mkdir -p .venvs || true
    [ -d ../.venvs/$name ]  && rm -Rf ../.venvs/$name
    python$pipver -m venv ../.venvs/$name
    ../.venvs/$name/bin/pip installl -U pip
    echo "[$(date)] Initialized Python virtual environment."
}

activate_venv () {
    VENV=$1
    . ../.venvs/$VENV/bin/activate
}
