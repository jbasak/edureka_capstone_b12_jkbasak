# --- CONFIGURATION ---
$envPath    = ".\pythonenv"
$reqFile    = "requirements.txt"
$commitMsg  = "Automated code sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Mute built-in PowerShell notification preferences 
$WarningPreference     = "SilentlyContinue"
$ProgressPreference    = "SilentlyContinue"
$InformationPreference = "SilentlyContinue"

try {
    # 1. REFRESH LOCAL CODES FROM GITHUB
    # Fetch and merge changes from the tracked remote branch silently
    git pull *> $null

    # 2. PYTHON VIRTUAL ENVIRONMENT MANAGEMENT
    # Check and build the local environment if missing
    if (-not (Test-Path "$envPath\Scripts\activate.ps1")) {
        python -m venv $envPath 1>$null
    }

    # Upgrade pip inside the localized environment container
    & "$envPath\Scripts\python.exe" -m pip install --upgrade pip 1>$null

    # Install packages if a dependencies document is found
    if (Test-Path $reqFile) {
        & "$envPath\Scripts\pip.exe" install -r $reqFile 1>$null
    }

    # Enable the Python environment workspace for the local console
    & "$envPath\Scripts\Activate.ps1"

    # 3. CHECK-IN NEW LOCAL CODES TO GITHUB
    # Stage all changes (new files, updates, deletions)
    git add -A *> $null

    # Commit the changes only if there is a delta to be checked in
    if ($(git status --porcelain)) {
        git commit -m $commitMsg *> $null
        git push *> $null
    }
}
catch {
    # If anything unexpected crashes, force out the error message
    Write-Error "Deployment script failed: $_"
    exit 1
}
