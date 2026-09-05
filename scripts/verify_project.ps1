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
    if (-not $SkipTests) {
        python -m pytest -q
    }
}
finally {
    Pop-Location
}
