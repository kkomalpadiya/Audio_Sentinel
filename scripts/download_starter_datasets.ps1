param(
    [string]$ProjectRoot = "C:\Users\kkoma\OneDrive\Desktop\Project_1",
    [string[]]$Datasets = @("esc50", "urbansound8k", "musan")
)

$rawRoot = Join-Path $ProjectRoot "data\raw"
$archiveRoot = Join-Path $rawRoot "_archives"

New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

function Download-File {
    param(
        [string]$Url,
        [string]$OutFile
    )

    if (Test-Path -LiteralPath $OutFile) {
        Write-Output "Skip existing archive: $OutFile"
        return
    }

    $partialFile = "$OutFile.partial"
    $existingBytes = if (Test-Path -LiteralPath $partialFile) {
        (Get-Item -LiteralPath $partialFile).Length
    } else {
        0
    }

    if ($existingBytes -gt 0) {
        Write-Output "Resuming from $existingBytes bytes: $Url"
    } else {
        Write-Output "Downloading: $Url"
    }

    # curl preserves the partial archive and resumes it with HTTP Range requests.
    # This is more reliable for multi-gigabyte files than Invoke-WebRequest.
    & curl.exe --fail --location --continue-at - --retry 8 --retry-all-errors `
        --retry-delay 5 --connect-timeout 30 --output $partialFile $Url

    if ($LASTEXITCODE -ne 0) {
        throw "Download interrupted. Keep '$partialFile' and run this script again to resume. curl exit code: $LASTEXITCODE"
    }

    Move-Item -LiteralPath $partialFile -Destination $OutFile
    Write-Output "Download complete: $OutFile"
}

function Ensure-CleanTarget {
    param([string]$TargetPath)

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null
    }
}

function Test-DatasetReady {
    param([string]$TargetPath)

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        return $false
    }

    # setup_dataset_dirs.ps1 creates empty placeholders. They are not datasets.
    return $null -ne (Get-ChildItem -LiteralPath $TargetPath -Recurse -File `
        -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gitkeep" } |
        Select-Object -First 1)
}

if ("esc50" -in $Datasets) {
    $zipPath = Join-Path $archiveRoot "esc50-master.zip"
    $target = Join-Path $rawRoot "esc50"
    $tmpExtract = Join-Path $rawRoot "esc50_extract_tmp"

    Download-File -Url "https://github.com/karolpiczak/esc-50/archive/refs/heads/master.zip" -OutFile $zipPath

    if (-not (Test-DatasetReady -TargetPath $target)) {
        if (Test-Path -LiteralPath $tmpExtract) {
            Remove-Item -LiteralPath $tmpExtract -Recurse -Force
        }

        New-Item -ItemType Directory -Force -Path $tmpExtract | Out-Null
        Expand-Archive -LiteralPath $zipPath -DestinationPath $tmpExtract -Force

        $extracted = Get-ChildItem -LiteralPath $tmpExtract -Directory | Select-Object -First 1
        if (-not $extracted) {
            throw "ESC-50 extraction failed."
        }

        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }

        Move-Item -LiteralPath $extracted.FullName -Destination $target
        Remove-Item -LiteralPath $tmpExtract -Recurse -Force
        Write-Output "Prepared: $target"
    } else {
        Write-Output "Skip existing dataset: $target"
    }
}

if ("urbansound8k" -in $Datasets) {
    $archivePath = Join-Path $archiveRoot "UrbanSound8K.tar.gz"
    $target = Join-Path $rawRoot "urbansound8k"
    $officialExtract = Join-Path $rawRoot "UrbanSound8K"

    Download-File -Url "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1" -OutFile $archivePath

    if (-not (Test-DatasetReady -TargetPath $target)) {
        tar -xzf $archivePath -C $rawRoot

        if (Test-Path -LiteralPath $officialExtract) {
            if (Test-Path -LiteralPath $target) {
                Remove-Item -LiteralPath $target -Recurse -Force
            }

            Move-Item -LiteralPath $officialExtract -Destination $target
        }

        Write-Output "Prepared: $target"
    } else {
        Write-Output "Skip existing dataset: $target"
    }
}

if ("musan" -in $Datasets) {
    $archivePath = Join-Path $archiveRoot "musan.tar.gz"
    $target = Join-Path $rawRoot "musan"

    Download-File -Url "https://openslr.trmal.net/resources/17/musan.tar.gz" -OutFile $archivePath

    if (-not (Test-DatasetReady -TargetPath $target)) {
        tar -xzf $archivePath -C $rawRoot
        Write-Output "Prepared: $target"
    } else {
        Write-Output "Skip existing dataset: $target"
    }
}

Write-Output "Starter dataset script finished."
