param(
    [string]$PythonExe = "",
    [switch]$SkipTask3
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$Requested)
    if ($Requested -and (Get-Command $Requested -ErrorAction SilentlyContinue)) {
        return $Requested
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "Python executable not found. Install Python or pass -PythonExe."
}

function Run-Step {
    param(
        [string]$PythonCmd,
        [string[]]$Args
    )
    Write-Host ">> $PythonCmd $($Args -join ' ')" -ForegroundColor Cyan
    & $PythonCmd @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $LASTEXITCODE"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

Push-Location $repoRoot
try {
    $python = Resolve-Python -Requested $PythonExe

    Write-Host "Constructing .pickle files from ZuCo .mat files..." -ForegroundColor Green
    Write-Host "Repository root: $repoRoot"
    Write-Host "Python: $python"

    Run-Step -PythonCmd $python -Args @(".\util\construct_dataset_mat_to_pickle_v1.py", "-t", "task1-SR")
    Run-Step -PythonCmd $python -Args @(".\util\construct_dataset_mat_to_pickle_v1.py", "-t", "task2-NR")
    if (-not $SkipTask3) {
        Run-Step -PythonCmd $python -Args @(".\util\construct_dataset_mat_to_pickle_v1.py", "-t", "task3-TSR")
    }
    Run-Step -PythonCmd $python -Args @(".\util\construct_dataset_mat_to_pickle_v2.py")

    Write-Host "Done. Pickle files are under .\dataset\ZuCo\*\pickle\" -ForegroundColor Green
}
finally {
    Pop-Location
}

