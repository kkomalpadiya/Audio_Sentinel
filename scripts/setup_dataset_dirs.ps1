$datasetDirs = @(
    "data/raw/fsd50k",
    "data/raw/urbansound8k",
    "data/raw/esc50",
    "data/raw/audioset_reference",
    "data/raw/musan",
    "data/raw/common_voice",
    "data/raw/ravdess",
    "data/raw/custom_threat_speech",
    "data/interim",
    "data/processed",
    "models"
)

foreach ($dir in $datasetDirs) {
    if (-not (Test-Path -LiteralPath $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Output "Created: $dir"
    } else {
        Write-Output "Exists: $dir"
    }
}

