<#
Sets up and starts Claude Code with the Fluent MCP server on the EPFL DLL VM.

Run it at the start of every VM session, from a normal PowerShell window:

    powershell -ExecutionPolicy Bypass -File "$([Environment]::GetFolderPath('MyDocuments'))\weisshorn-nose-cone\CFD\mcp\bootstrap-vm.ps1"

Each step is skipped when its result is already there. The first run installs everything (about
10 minutes); later runs take seconds, also after C:\ has been wiped. It follows
2026_C_FD_REPORT-2_MCP-FLUENT (Claude Code track) with two differences:

* the MCP server is CFD\mcp\server.py of this repository (the guide clones it from a repo whose
  URL it does not give);
* the Claude Code home is moved to Z:\ with CLAUDE_CONFIG_DIR instead of a junction: a junction
  cannot point to a network drive, so the guide's mklink /J to Z:\ does not work.

AppLocker blocks executables and native Python modules on Z:\, so claude.exe and the Python
packages live on C:\. The network home (Z:\, or the Documents folder when there is no Z:\) keeps
the golden copies (tools\), the login (home\.claude) and this repository.
#>
$ErrorActionPreference = "Stop"

$Repo       = (Resolve-Path "$PSScriptRoot\..\..").Path
$Server     = "$PSScriptRoot\server.py"
# Persistent storage: the network home Z:\ of the guide, else the Documents folder (on these PCs it
# may be redirected to the network home instead of mapped as Z:\).
$Persist    = if (Test-Path "Z:\") { "Z:\" } else { [Environment]::GetFolderPath("MyDocuments") }
$Wheels     = Join-Path $Persist "tools\wheels"
$ClaudeGold = Join-Path $Persist "tools\claude-code"
$ClaudeHome = Join-Path $Persist "home\.claude"
$ClaudeExe  = "$env:USERPROFILE\.local\bin\claude.exe"   # where the native installer puts it
# The versions server.py is written for. mcp 2 renamed FastMCP, and later PyFluent releases drop
# the oldest Fluent versions, so the guide's open ranges (>=0.30, >=1.2) are pinned here.
$Packages   = @("ansys-fluent-core>=0.43,<0.44", "mcp>=2.3,<3")

function Step($Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }

# Runs a native command and returns $true if it exits with 0, without letting its stderr stop
# the script (Windows PowerShell 5.1 turns redirected stderr into errors).
function Test-Native([scriptblock]$Block) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) }
    finally { $ErrorActionPreference = $old }
}

Step "Persistent storage: $Persist"
if ($Persist -like "C:*") {
    Write-Host "Warning: no network home found. $Persist is on C:\ and goes with the profile purge." -ForegroundColor Yellow
}
New-Item -ItemType Directory -Force $Wheels, $ClaudeGold, $ClaudeHome, "$Repo\CFD\runs" | Out-Null

# --- Fluent --------------------------------------------------------------------------------------
$Awp = Get-ChildItem Env: | Where-Object { $_.Name -match '^AWP_ROOT\d{3}$' } |
    Sort-Object Name | Select-Object -Last 1
if ($Awp) {
    $AwpName = $Awp.Name; $AwpRoot = $Awp.Value
} else {
    $dir = Get-ChildItem "C:\Program Files\ANSYS Inc" -Directory -Filter "v2*" -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1
    if (-not $dir) { throw "No Ansys installation found (no AWP_ROOTxxx variable, nothing in C:\Program Files\ANSYS Inc)." }
    $AwpName = "AWP_ROOT" + $dir.Name.Substring(1); $AwpRoot = $dir.FullName
}
Step "Fluent: $AwpName = $AwpRoot"

# --- Python and the packages (C:\, pip --user, from the wheel cache on Z:\) -----------------------
$Python = $null
foreach ($candidate in @("C:\Program Files\Python313\python.exe", "C:\Program Files\Python312\python.exe")) {
    if (-not $Python -and (Test-Path $candidate)) { $Python = $candidate }
}
if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") { $Python = $cmd.Source }
}
if (-not $Python) {
    # Last resort: the Python 3.10 that ships with Ansys
    $Python = Get-ChildItem "$AwpRoot\commonfiles\CPython\*\winx64\Release\python\python.exe" -ErrorAction SilentlyContinue |
        Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $Python) { throw "Python 3.10+ not found. The guide expects C:\Program Files\Python313\python.exe." }
Step "Python: $Python"

$check = "import importlib.metadata as m, sys; sys.exit(not (m.version('ansys-fluent-core').startswith('0.43.') and m.version('mcp').startswith('2.')))"
if (-not (Test-Native { & $Python -c $check })) {
    Step "Installing PyFluent and mcp on C:\ (pip --user)"
    if (-not (Test-Native { & $Python -m pip download --dest $Wheels @Packages --quiet })) {
        Write-Host "pip download failed (no internet?). Using the wheels already in $Wheels."
    }
    $env:PYTHONUSERBASE = "$env:APPDATA\Python"
    & $Python -m pip install --user --no-index --find-links $Wheels @Packages --quiet
    if ($LASTEXITCODE -ne 0) { throw "pip install failed. Check that $Wheels holds the wheels." }
}
& $Python -c "import ansys.fluent.core as f, importlib.metadata as m; print('PyFluent', f.__version__, '| mcp', m.version('mcp'))"

# --- Claude Code (binary on C:\, golden copy and login on Z:\) ------------------------------------
if (-not (Test-Path $ClaudeExe)) {
    if (Test-Path "$ClaudeGold\claude.exe") {
        Step "Restoring claude.exe from $ClaudeGold"
        New-Item -ItemType Directory -Force (Split-Path $ClaudeExe) | Out-Null
        Copy-Item "$ClaudeGold\claude.exe" $ClaudeExe
    } else {
        Step "Installing Claude Code (native installer)"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-RestMethod https://claude.ai/install.ps1 | Invoke-Expression
        if (-not (Test-Path $ClaudeExe)) { throw "The installer did not create $ClaudeExe. Use method A (npm) of the guide, Section 3.4.1." }
    }
}
Copy-Item $ClaudeExe "$ClaudeGold\claude.exe" -Force   # keeps the golden copy at the auto-updated version
$env:PATH = "$(Split-Path $ClaudeExe);$env:PATH"
$env:CLAUDE_CONFIG_DIR = $ClaudeHome
[Environment]::SetEnvironmentVariable("CLAUDE_CONFIG_DIR", $ClaudeHome, "User")
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Warning: git is not on PATH. Claude Code on Windows needs Git for Windows (Git Bash)." -ForegroundColor Yellow
}
Write-Host "Claude Code $(& $ClaudeExe --version), home $ClaudeHome"

# --- MCP server registration: project .mcp.json, UTF-8 without BOM (not tracked by git) -----------
$config = [ordered]@{
    mcpServers = [ordered]@{
        fluent = [ordered]@{
            command = $Python
            args    = @($Server)
            env     = [ordered]@{ $AwpName = $AwpRoot; PYFLUENT_LAUNCH_CONTAINER = "0" }
        }
    }
}
[IO.File]::WriteAllText("$Repo\.mcp.json", ($config | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
Write-Host "Wrote $Repo\.mcp.json"

# --- Start -----------------------------------------------------------------------------------------
$env:MCP_TIMEOUT = "60000"         # PyFluent takes a while to import
$env:MCP_TOOL_TIMEOUT = "3600000"  # long iterate calls
$runs = "$Repo\CFD\runs" -replace '\\', '/'
Step "Starting Claude Code in $Repo"
Write-Host "First run: sign in in the browser, trust the folder, approve the 'fluent' server (/mcp)."
Write-Host "In Fluent, after reading the case:  (chdir `"$runs`")  then  server/start-server  (Enter)"
Write-Host "Then ask Claude to connect_to_fluent with $runs"
Set-Location $Repo
& $ClaudeExe
