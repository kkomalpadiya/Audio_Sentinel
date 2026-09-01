param(
    [string]$ProjectRoot = "C:\Users\kkoma\OneDrive\Desktop\Project_1",
    [int]$ChunkSizeMB = 16
)

$url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
$totalBytes = 6023741708L
$archiveRoot = Join-Path $ProjectRoot "data\raw\_archives"
$partialPath = Join-Path $archiveRoot "UrbanSound8K.tar.gz.partial"
$archivePath = Join-Path $archiveRoot "UrbanSound8K.tar.gz"
$rawRoot = Join-Path $ProjectRoot "data\raw"
$target = Join-Path $rawRoot "urbansound8k"
$chunkBytes = [int64]$ChunkSizeMB * 1MB
$chunkPath = "$partialPath.chunk"

if ($ChunkSizeMB -lt 1) {
    throw "ChunkSizeMB must be at least 1."
}

New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

if (-not (Test-Path -LiteralPath $archivePath)) {
    if (-not (Test-Path -LiteralPath $partialPath)) {
        New-Item -ItemType File -Path $partialPath | Out-Null
    }

    $offset = (Get-Item -LiteralPath $partialPath).Length
    if ($offset -gt $totalBytes) {
        throw "The partial archive is larger than the official file. Move it aside and start again."
    }

    while ($offset -lt $totalBytes) {
        $end = [Math]::Min($offset + $chunkBytes - 1, $totalBytes - 1)
        $expectedBytes = $end - $offset + 1
        Remove-Item -LiteralPath $chunkPath -Force -ErrorAction SilentlyContinue

        Write-Output "Downloading bytes $offset through $end of $totalBytes"
        & curl.exe --fail --location --range "$offset-$end" --retry 4 --retry-all-errors `
            --retry-delay 5 --connect-timeout 30 --max-time 600 --output $chunkPath $url

        if ($LASTEXITCODE -ne 0) {
            throw "Chunk download interrupted. The archive is unchanged; run this script again to retry this chunk. curl exit code: $LASTEXITCODE"
        }

        $receivedBytes = (Get-Item -LiteralPath $chunkPath).Length
        if ($receivedBytes -ne $expectedBytes) {
            throw "Chunk size check failed. Expected $expectedBytes bytes but received $receivedBytes. Run the script again to retry this chunk."
        }

        $source = [System.IO.File]::OpenRead($chunkPath)
        $destination = [System.IO.File]::Open($partialPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
        try {
            $source.CopyTo($destination)
        } finally {
            $destination.Dispose()
            $source.Dispose()
        }

        Remove-Item -LiteralPath $chunkPath -Force
        $offset += $receivedBytes
    }

    Move-Item -LiteralPath $partialPath -Destination $archivePath
    Write-Output "Archive complete: $archivePath"
}

if (-not (Test-Path -LiteralPath $target) -or
    $null -eq (Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".gitkeep" } | Select-Object -First 1)) {
    tar -xzf $archivePath -C $rawRoot
}

Write-Output "UrbanSound8K is ready at: $target"
