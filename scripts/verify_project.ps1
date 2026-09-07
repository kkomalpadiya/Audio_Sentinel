[CmdletBinding()]
param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Push-Location $projectRoot
try {
    python -m compileall -q src
    if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }
    if (-not $SkipTests) {
        python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "Project tests failed." }
        python scripts/smoke_test_preparation.py
        if ($LASTEXITCODE -ne 0) { throw "Preparation smoke test failed." }
    }
}
finally {
    Pop-Location
}
