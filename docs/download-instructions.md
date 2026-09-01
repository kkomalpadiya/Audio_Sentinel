# Dataset Download Instructions

This guide explains exactly what to download first, where to put it, and which datasets to postpone.

## Recommended order

### Download now

1. ESC-50
2. UrbanSound8K
3. MUSAN

### Download later

1. FSD50K
2. Common Voice
3. RAVDESS

## Why this order

- `ESC-50` is small enough to start quickly.
- `UrbanSound8K` gives useful `gun_shot` and `siren` classes early.
- `MUSAN` helps preprocessing, speech-vs-noise checks, and augmentation.
- `FSD50K` is large and split across multiple archives, so it should be staged after the first batch is in place.

## Storage note

On August 30, 2026, the machine has about 102 GB free on drive `C:`.

That is enough for the starter batch, but it is not comfortable for downloading and extracting everything at once if you include `FSD50K`. Download `FSD50K` only after the first batch is organized and you confirm you still have enough space.

## Folder targets

Download and extract only into:

- `data/raw/esc50`
- `data/raw/urbansound8k`
- `data/raw/musan`
- `data/raw/fsd50k`

## Option A: use the helper scripts

From `C:\Users\kkoma\OneDrive\Desktop\Project_1`, run:

```powershell
cd C:\Users\kkoma\OneDrive\Desktop\Project_1
.\scripts\download_starter_datasets.ps1
```

That script downloads and extracts:

- `ESC-50`
- `UrbanSound8K`
- `MUSAN`

It uses Windows `curl` to resume large downloads. If your connection stops, do
not delete the `.partial` file in `data/raw/_archives`; run the same command
again and it will continue from the downloaded size.

If Zenodo repeatedly drops the long UrbanSound8K connection, use the chunked
resume helper instead. It makes short 16 MB range requests and preserves the
existing `UrbanSound8K.tar.gz.partial` file:

```powershell
.\scripts\resume_urbansound8k_chunked.ps1
```

For MUSAN, use the equivalent chunked downloader:

```powershell
.\scripts\download_musan_chunked.ps1
```

If OneDrive is actively syncing while a download runs, pause OneDrive temporarily
or download to a non-OneDrive drive with enough free space. Dataset files are
ignored by Git, so they do not need to be committed.

For `FSD50K`, use:

```powershell
.\scripts\download_fsd50k_parts.ps1
```

That second script downloads the official archive parts into `data/raw/fsd50k/archives/`, but does not extract them automatically.

## Option B: manual download steps

### 1. ESC-50

Official page:

- <https://github.com/karolpiczak/esc-50>

Direct archive:

- <https://github.com/karolpiczak/esc-50/archive/refs/heads/master.zip>

What to do:

1. Download the zip.
2. Extract it.
3. Rename the extracted folder to `esc50`.
4. Place it under `data/raw/`.

Final path:

- `data/raw/esc50`

### 2. UrbanSound8K

Official page:

- <https://urbansounddataset.weebly.com/urbansound8k.html>

Direct download:

- <https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1>

What to do:

1. Download `UrbanSound8K.tar.gz`.
2. Extract it.
3. If the extracted folder is named `UrbanSound8K`, rename it to `urbansound8k`.
4. Place it under `data/raw/`.

Final path:

- `data/raw/urbansound8k`

### 3. MUSAN

Official page:

- <https://www.openslr.org/17/>

Direct mirror:

- <https://openslr.trmal.net/resources/17/musan.tar.gz>

What to do:

1. Download `musan.tar.gz`.
2. Extract it.
3. Keep the extracted folder name as `musan`.
4. Place it under `data/raw/`.

Final path:

- `data/raw/musan`

### 4. FSD50K

Official companion page:

- <https://fsannotator.upf.edu/fsd/release/FSD50K/>

Official Zenodo record:

- <https://zenodo.org/records/4060432>

Required files:

- `FSD50K.dev_audio.zip`
- `FSD50K.dev_audio.z01`
- `FSD50K.dev_audio.z02`
- `FSD50K.dev_audio.z03`
- `FSD50K.dev_audio.z04`
- `FSD50K.dev_audio.z05`
- `FSD50K.eval_audio.zip`
- `FSD50K.eval_audio.z01`
- `FSD50K.ground_truth.zip`
- `FSD50K.metadata.zip`
- `FSD50K.doc.zip`

What to do:

1. Download all required parts into `data/raw/fsd50k/archives/`.
2. Merge and extract the split zip files with a tool that supports multi-part zip archives.
3. Extract the smaller metadata archives in the same dataset root.

Expected extracted folders:

- `FSD50K.dev_audio`
- `FSD50K.eval_audio`
- `FSD50K.ground_truth`
- `FSD50K.metadata`
- `FSD50K.doc`

Final path:

- `data/raw/fsd50k`

## Recommended next action

Run the starter download script first. After those datasets are in place, we can:

1. verify the folder layout
2. add a metadata manifest for citations and licenses
3. build the first offline ingestion script
