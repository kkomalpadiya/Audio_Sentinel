param(
    [string]$ProjectRoot = "C:\Users\kkoma\OneDrive\Desktop\Project_1"
)

$archiveRoot = Join-Path $ProjectRoot "data\raw\fsd50k\archives"
New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

$files = @(
    "FSD50K.dev_audio.zip",
    "FSD50K.dev_audio.z01",
    "FSD50K.dev_audio.z02",
    "FSD50K.dev_audio.z03",
    "FSD50K.dev_audio.z04",
    "FSD50K.dev_audio.z05",
    "FSD50K.eval_audio.zip",
    "FSD50K.eval_audio.z01",
    "FSD50K.ground_truth.zip",
    "FSD50K.metadata.zip",
    "FSD50K.doc.zip"
)

foreach ($file in $files) {
    $url = "https://zenodo.org/record/4060432/files/$file?download=1"
    $outFile = Join-Path $archiveRoot $file

    if (Test-Path -LiteralPath $outFile) {
        Write-Output "Skip existing archive: $outFile"
        continue
    }

    Write-Output "Downloading: $file"
    Invoke-WebRequest -Uri $url -OutFile $outFile
}

Write-Output "FSD50K archive download finished."
Write-Output "Next step: extract the multipart archives with a tool that supports split zip files."

