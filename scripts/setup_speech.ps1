param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentDirectory = Join-Path $projectRoot ".venv\speech"
$runtimePython = Join-Path $environmentDirectory "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $Python -m venv $environmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the speech virtual environment." }
}

& $runtimePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the speech environment." }

& $runtimePython -m pip install -e "${projectRoot}[speech]"
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned speech runtime." }

& $runtimePython (Join-Path $PSScriptRoot "install_silero_vad.py")
if ($LASTEXITCODE -ne 0) { throw "Could not install or verify Silero VAD v6." }

& $runtimePython (Join-Path $PSScriptRoot "smoke_test_vad.py")
if ($LASTEXITCODE -ne 0) { throw "The Silero VAD smoke test failed." }

& $runtimePython (Join-Path $PSScriptRoot "smoke_test_speech_segments.py")
if ($LASTEXITCODE -ne 0) { throw "The speech-segment extraction smoke test failed." }
