$envPath = ".\pythonenv"
$reqFile = "requirements.txt"

# Force PowerShell preferences to suppress everything except errors
$WarningPreference     = "SilentlyContinue"
$ProgressPreference    = "SilentlyContinue"
$InformationPreference = "SilentlyContinue"

# 1. Check and create the Python virtual environment
if (-not (Test-Path "$envPath\Scripts\activate.ps1")) {
    Write-Host "Creating Python environment..." -ForegroundColor Cyan
    python -m venv $envPath 1>$null
} else {
    Write-Host "Environment already exists. Skipping creation." -ForegroundColor Yellow
}

# 2. Upgrade pip inside the local environment
Write-Host "Updating pip to latest version..." -ForegroundColor Cyan
& "$envPath\Scripts\python.exe" -m pip install --upgrade pip 1>$null

# 3. Check and install dependencies from requirements.txt
if (Test-Path $reqFile) {
    Write-Host "Installing dependencies from $reqFile..." -ForegroundColor Cyan
    & "$envPath\Scripts\pip.exe" install -r $reqFile 1>$null
} else {
    Write-Host "No $reqFile found. Skipping package installation." -ForegroundColor Yellow
}

# 4. Activate the environment for the current session
Write-Host "Activating Python environment..." -ForegroundColor Cyan
& "$envPath\Scripts\Activate.ps1"
cd C:\BASAK\Codebase\github_repos\edureka_capstone_b12_jkbasak
