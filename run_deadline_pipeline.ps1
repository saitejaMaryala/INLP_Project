param(
    [string]$PythonExe = ".\\.venv\\Scripts\\python.exe",
    [switch]$SkipKill
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )

    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $Action
}

function Ensure-Path {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Required path not found: $Path"
    }
}

Run-Step "Set project root" {
    Set-Location "C:\Users\monos\INLP\INLP_after_mid\INLP_Project"
    Write-Host "Working directory: $(Get-Location)"
}

Run-Step "Optional cleanup of running platt jobs" {
    if ($SkipKill) {
        Write-Host "Skipping process cleanup (--SkipKill enabled)."
        return
    }

    $procs = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -match "training.compare_quant_bios" -and
            $_.CommandLine -match "score-calibration platt"
        }

    if (-not $procs) {
        Write-Host "No running platt calibration jobs found."
        return
    }

    foreach ($p in $procs) {
        Write-Host "Stopping PID $($p.ProcessId): $($p.CommandLine)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Run-Step "Create timestamped archive folder" {
    $script:Stamp = Get-Date -Format "yyyyMMdd_HHmm"
    $script:ArchiveBase = Join-Path "results\\archive" $script:Stamp
    New-Item -ItemType Directory -Force -Path $script:ArchiveBase | Out-Null
    Write-Host "Archive base: $script:ArchiveBase"
}

Run-Step "Archive current Bias results" {
    Ensure-Path "results\\bios"
    Copy-Item -Recurse -Force "results\\bios" (Join-Path $script:ArchiveBase "bias_current")
}

Run-Step "Archive current Jigsaw phase23 as original FROC" {
    Ensure-Path "outputs\\phase23"
    Copy-Item -Recurse -Force "outputs\\phase23" (Join-Path $script:ArchiveBase "jigsaw_original_froc")
}

Run-Step "Run Jigsaw with current algorithm" {
    Ensure-Path $PythonExe
    & $PythonExe -m pretrained.unbert_ju.compare_quantized
}

Run-Step "Archive new Jigsaw phase23 as current FROC" {
    Ensure-Path "outputs\\phase23"
    Copy-Item -Recurse -Force "outputs\\phase23" (Join-Path $script:ArchiveBase "jigsaw_current_froc")
}

Run-Step "Generate Jigsaw report diff" {
    $orig = Join-Path $script:ArchiveBase "jigsaw_original_froc\\phase23_verification_report.md"
    $curr = Join-Path $script:ArchiveBase "jigsaw_current_froc\\phase23_verification_report.md"
    Ensure-Path $orig
    Ensure-Path $curr

    $diffOut = Join-Path $script:ArchiveBase "jigsaw_froc_diff.txt"
    Compare-Object (Get-Content $orig) (Get-Content $curr) |
        Out-File -FilePath $diffOut -Encoding UTF8

    Write-Host "Diff written: $diffOut"
}

Run-Step "Final deliverable checks" {
    $checks = @(
        "results\\bios\\phase23\\phase23_verification_report.md",
        "results\\bios\\phase23\\transport_diagnostics.json",
        "results\\bios\\phase23\\threshold_invariance.csv",
        "outputs\\phase23\\phase23_verification_report.md",
        (Join-Path $script:ArchiveBase "jigsaw_froc_diff.txt")
    )

    foreach ($p in $checks) {
        $ok = Test-Path $p
        Write-Host ("{0} : {1}" -f $p, $(if ($ok) { "OK" } else { "MISSING" }))
    }
}

Write-Host "`nDone. Archive: $script:ArchiveBase" -ForegroundColor Green
