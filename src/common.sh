
init_venv () {
    project_root="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
    name="$1"
    pythonver="$2"
    pipver="$3"
    #mkdir -p "$project_root/.venvs" || true
    #[ -d "$project_root/.venvs/$name" ]  && rm -Rf $project_root/.venvs/$name
    if [ ! -d "$project_root/.venvs/$name" ]; then
        echo "[$(date)] Initializing Python virtual environment '$name' using Python $pythonver and pip $pipver."
        mkdir -p "$project_root/.venvs" || true
        python$pipver -m venv $project_root/.venvs/$name
        $project_root/.venvs/$name/bin/pip install -U pip
        echo "[$(date)] Initialized Python virtual environment."
    fi
}

activate_venv () {
    project_root="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/.."
    VENV=$1
    . $project_root/.venvs/$VENV/bin/activate
}
