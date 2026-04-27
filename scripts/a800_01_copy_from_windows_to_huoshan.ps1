param(
    [string]$RemoteHost = "HuoShan2",
    [string]$RemoteRoot = "/vePFS-0x0d/nzh",
    [string]$SrcRepo = "\\10.16.93.90\dataset3\nzh\lcm_ds\brain_network_decoder",
    [string]$SrcPretrain = "D:\NCClab\LCM_fork_hvr710\pretrain_weights_fold0",
    [string]$SrcDataset1 = "\\10.16.57.94\dataset1",
    [string]$SrcDataset4 = "\\10.20.33.82\dataset4",
    [switch]$CopyRepo
)

$ErrorActionPreference = "Stop"

$RunTs = Get-Date -Format "yyyyMMdd_HHmmss"
$RemoteRootWin = $RemoteRoot.Replace("/", "\")
$RemoteLeaf = [System.IO.Path]::GetFileName($RemoteRootWin)
$RemoteParentWin = [System.IO.Path]::GetDirectoryName($RemoteRootWin)
$RemoteParent = $RemoteParentWin.Replace("\", "/")

$StageRoot = Join-Path $env:TEMP "a800_table3_stage_$RunTs"
$StageTop = Join-Path $StageRoot $RemoteLeaf
$StageRepo = Join-Path $StageTop "repo\brain_network_decoder"
$StageData = Join-Path $StageTop "data"
$StageRoi = Join-Path $StageData "dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi"
$StageLabel = Join-Path $StageData "dataset1\ningzh\labels"

$RepoDst = "$RemoteRoot/repo/brain_network_decoder"
$DataDst = "$RemoteRoot/data"
$OneRoot = "outputs/one_$RunTs"
$SrcRoi = Join-Path $SrcDataset4 "DATASETS\fmri_pretraining\fmri_dataset\roi"
$SrcLabel = Join-Path $SrcDataset1 "ningzh\labels"

function Require-Path([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing path: $Path"
    }
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function New-Junction([string]$LinkPath, [string]$TargetPath) {
    Require-Path $TargetPath
    if (Test-Path -LiteralPath $LinkPath) {
        Remove-Item -LiteralPath $LinkPath -Force -Recurse
    }
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
}

function Copy-DirLocal([string]$Source, [string]$Dest) {
    Require-Path $Source
    Ensure-Dir $Dest
    & robocopy $Source $Dest /E /R:2 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for $Source -> $Dest (exit $LASTEXITCODE)"
    }
}

function Copy-FileLocal([string]$Source, [string]$DestDir) {
    Require-Path $Source
    Ensure-Dir $DestDir
    Copy-Item -LiteralPath $Source -Destination $DestDir -Force
}

function Start-UploadSession([string]$HostName, [string]$RemoteParentDir) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "ssh"
    $psi.Arguments = "$HostName ""mkdir -p '$RemoteParentDir' && tar -xf - -C '$RemoteParentDir'"""
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $false
    $psi.RedirectStandardError = $false
    $psi.CreateNoWindow = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    return $proc
}

function Start-TarStream([string]$WorkingDir, [string]$TopName) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "tar"
    $psi.Arguments = "-chf - $TopName"
    $psi.WorkingDirectory = $WorkingDir
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $false
    $psi.CreateNoWindow = $false

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    return $proc
}

Write-Host "==> Opening upload session to $RemoteHost now"
Write-Host "==> Please enter the SSH key passphrase once when prompted"
$UploadProc = Start-UploadSession $RemoteHost $RemoteParent

Write-Host "==> Staging files locally under $StageTop"
Ensure-Dir $StageTop
Ensure-Dir $StageRepo
Ensure-Dir $StageData

$EnvFile = Join-Path $StageTop "run_table3_one.env"
@"
export A800_ROOT=$RemoteRoot
export REPO_DST=$RepoDst
export DATA_DST=$DataDst
export RUN_TS=$RunTs
export ONE_ROOT=$OneRoot
export TABLE3_DATASET1_ROOT=$DataDst/dataset1
export TABLE3_DATASET3_ROOT=$DataDst/dataset3
export TABLE3_DATASET4_ROOT=$DataDst/dataset4
"@ | Set-Content -LiteralPath $EnvFile -Encoding ascii

if ($CopyRepo) {
    Write-Host "==> Staging repo"
    Copy-DirLocal $SrcRepo $StageRepo
} else {
    Write-Host "==> Skipping repo copy (repo should already be cloned on HuoShan)"
}

Write-Host "==> Wiring pretrained weights from local path without recopy"
New-Junction (Join-Path $StageRepo "pretrain_weights_fold0") $SrcPretrain

Write-Host "==> Staging dataset4 Table3 ROI data and splits"
Copy-DirLocal (Join-Path $SrcRoi "ABIDE\AAL") (Join-Path $StageRoi "ABIDE\AAL")
Copy-DirLocal (Join-Path $SrcRoi "ABIDE\Schaefer2018_100_crop_split") (Join-Path $StageRoi "ABIDE\Schaefer2018_100_crop_split")

Copy-DirLocal (Join-Path $SrcRoi "NKI\AAL") (Join-Path $StageRoi "NKI\AAL")
Copy-DirLocal (Join-Path $SrcRoi "NKI\100ROI_split") (Join-Path $StageRoi "NKI\100ROI_split")

Copy-DirLocal (Join-Path $SrcRoi "SALD\AAL") (Join-Path $StageRoi "SALD\AAL")
Copy-DirLocal (Join-Path $SrcRoi "SALD\100ROI_split") (Join-Path $StageRoi "SALD\100ROI_split")

Copy-DirLocal (Join-Path $SrcRoi "ABCD\AAL") (Join-Path $StageRoi "ABCD\AAL")
Copy-DirLocal (Join-Path $SrcRoi "HCP\AAL") (Join-Path $StageRoi "HCP\AAL")

Copy-DirLocal (Join-Path $SrcRoi "BHRC\AAL") (Join-Path $StageRoi "BHRC\AAL")
Copy-DirLocal (Join-Path $SrcRoi "BHRC\100ROI_split") (Join-Path $StageRoi "BHRC\100ROI_split")

Copy-DirLocal (Join-Path $SrcRoi "PPMI\AAL") (Join-Path $StageRoi "PPMI\AAL")
Copy-DirLocal (Join-Path $SrcRoi "PPMI\100ROI") (Join-Path $StageRoi "PPMI\100ROI")

Copy-DirLocal (Join-Path $SrcRoi "ADNI\AAL\CN") (Join-Path $StageRoi "ADNI\AAL\CN")
Copy-DirLocal (Join-Path $SrcRoi "ADNI\AAL\MCI") (Join-Path $StageRoi "ADNI\AAL\MCI")
Copy-DirLocal (Join-Path $SrcRoi "ADNI\AAL\AD") (Join-Path $StageRoi "ADNI\AAL\AD")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\MCI_train_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\MCI_val_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\MCI_test_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\AD_train_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\AD_val_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")
Copy-FileLocal (Join-Path $SrcRoi "ADNI\AAL\AD_test_abs_AAL.txt") (Join-Path $StageRoi "ADNI\AAL")

Write-Host "==> Staging dataset1 labels"
Copy-FileLocal (Join-Path $SrcLabel "age\ABIDE.csv") (Join-Path $StageLabel "age")
Copy-FileLocal (Join-Path $SrcLabel "age\NKI.csv") (Join-Path $StageLabel "age")
Copy-FileLocal (Join-Path $SrcLabel "age\SALD.csv") (Join-Path $StageLabel "age")

Copy-FileLocal (Join-Path $SrcLabel "sex\ABCD.csv") (Join-Path $StageLabel "sex")
Copy-FileLocal (Join-Path $SrcLabel "sex\HCP.csv") (Join-Path $StageLabel "sex")
Copy-FileLocal (Join-Path $SrcLabel "sex\BHRC.csv") (Join-Path $StageLabel "sex")

Copy-FileLocal (Join-Path $SrcLabel "disease\PPMI.csv") (Join-Path $StageLabel "disease")
Copy-FileLocal (Join-Path $SrcLabel "disease\adni_list.xlsx") (Join-Path $StageLabel "disease")

Copy-FileLocal (Join-Path $SrcLabel "education\NKI.csv") (Join-Path $StageLabel "education")

Write-Host "==> Approx local staging size"
Get-ChildItem -LiteralPath $StageTop -Recurse -File | Measure-Object -Property Length -Sum | ForEach-Object {
    "{0:N2} GB" -f ($_.Sum / 1GB)
}

Write-Host "==> Packing staged tree and streaming it over the existing SSH session"
$TarProc = Start-TarStream $StageRoot $RemoteLeaf
try {
    $TarProc.StandardOutput.BaseStream.CopyTo($UploadProc.StandardInput.BaseStream)
}
finally {
    $TarProc.StandardOutput.Close()
    $UploadProc.StandardInput.Close()
}

$TarProc.WaitForExit()
if ($TarProc.ExitCode -ne 0) {
    throw "tar stream failed with exit code $($TarProc.ExitCode)"
}

$UploadProc.WaitForExit()
if ($UploadProc.ExitCode -ne 0) {
    throw "remote upload/extract failed with exit code $($UploadProc.ExitCode)"
}

Write-Host "==> Cleaning local staging directory"
Remove-Item -LiteralPath $StageRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "==> Done"
Write-Host "==> Remote root: $RemoteRoot"
Write-Host "==> Next on HuoShan:"
Write-Host "source $RemoteRoot/run_table3_one.env"
Write-Host "cd $RepoDst"
Write-Host "bash scripts/a800_run_step_in_tmux.sh env_1 scripts/a800_02_create_envs.sh"
