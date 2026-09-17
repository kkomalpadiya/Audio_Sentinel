param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentDirectory = Join-Path $projectRoot ".venv\yamnet"
$runtimePython = Join-Path $environmentDirectory "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $Python -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the YAMNet virtual environment." }
}

& $runtimePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the YAMNet environment." }

& $runtimePython -m pip install -e "${projectRoot}[yamnet]"
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned YAMNet runtime." }

& $runtimePython (Join-Path $PSScriptRoot "download_yamnet_model.py")
if ($LASTEXITCODE -ne 0) { throw "Could not download or verify YAMNet v1." }

& $runtimePython (Join-Path $PSScriptRoot "smoke_test_acoustic_loader.py")
if ($LASTEXITCODE -ne 0) { throw "The YAMNet loader smoke test failed." }
