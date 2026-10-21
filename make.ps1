<#
make.ps1 — Windows entry points, semantic parity with the Makefile (workorder G27).

Usage:
    .\make.ps1 <target> [-Port 8093]

Targets (same names and semantics as `make`):
    help | setup | lint | test | serve | smoke | bench | fetch-data
    docker-build | docker-serve | docker-smoke | docker-stop

Notes:
    * Makefile stays authoritative on POSIX; this script must stay a thin
      wrapper with identical target semantics (Phase A deliverable A4).
    * Each smoke/bench run boots a throwaway service over its own temp db,
      exactly like the Makefile's mktemp -d flow.
#>
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [int]$Port = 8093
)

$ErrorActionPreference = "Stop"
$Py   = ".venv\Scripts\python.exe"
$Pip  = ".venv\Scripts\pip.exe"
$Base = "http://127.0.0.1:$Port"

function New-TempDb([string]$Kind) {
    # temp paths on Windows user profiles contain no spaces for the common
    # case; if yours does, set TMPDIR/TEMP to a space-free directory
    return Join-Path ([System.IO.Path]::GetTempPath()) ("aml_{0}_{1}.db" -f $Kind, [guid]::NewGuid().ToString("N"))
}

function Start-AmlServer([string]$DbPath) {
    $proc = Start-Process -FilePath $Py `
        -ArgumentList @("serve.py", "--port", "$Port", "--set", "storage.db_path=$DbPath") `
        -PassThru -NoNewWindow
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri "$Base/health" -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) { return $proc }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    try { $proc.Kill() } catch {}
    throw "service did not become healthy on $Base"
}

function Invoke-WithServer([string]$Kind, [scriptblock]$Body) {
    $proc = Start-AmlServer (New-TempDb $Kind)
    try { & $Body }
    finally {
        try { $proc.Kill(); $proc.WaitForExit(5000) | Out-Null } catch {}
    }
}

function Remove-AppleDouble {
    Get-ChildItem -Recurse -Force -Filter "._*" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\.venv\\|\\.git\\' } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

switch ($Target) {
    "help" {
        Write-Host "targets: setup lint test serve smoke bench docker-build docker-serve docker-smoke docker-stop fetch-data"
    }
    "setup" {
        if (-not (Test-Path ".venv")) {
            # prefer 3.14 (the pin the Mac baseline used); fall back to the
            # launcher default, mirroring the Makefile's plain `python3`
            py -3.14 -m venv .venv 2>$null
            if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv 2>$null }
            if (-not (Test-Path ".venv")) { py -m venv .venv }
        }
        & $Pip install -q --upgrade pip
        & $Pip install -q -r requirements.lock
        & $Pip install -q -e . --no-deps
        Write-Host "setup done"
    }
    "lint" {
        & $Py -m ruff check .
    }
    "test" {
        & $Py -m pytest 08_tests
    }
    "serve" {
        & $Py serve.py --port $Port
    }
    "smoke" {
        Invoke-WithServer "smoke" { & $Py smoke.py --base-url $Base --wait 30 }
    }
    "bench" {
        if (-not (Test-Path ".venv")) { & .\make.ps1 setup } else { & $Pip install -q -e . --no-deps }
        if (-not (Test-Path "04_bench\data\locomo10.json")) { & $Py 04_bench\fetch_locomo.py }
        Invoke-WithServer "bench" {
            & $Py smoke.py --base-url $Base --wait 30
            if ($LASTEXITCODE -ne 0) { throw "smoke failed" }
            & $Py 04_bench\run_bench.py --base-url $Base --label local-baseline
        }
    }
    "fetch-data" {
        & $Py 04_bench\fetch_locomo.py
    }
    "docker-build" {
        Remove-AppleDouble
        docker build -t zhiji-aml:v0 -f 05_docker\Dockerfile .
    }
    "docker-serve" {
        docker compose -f 05_docker\compose.yml up -d --build
    }
    "docker-smoke" {
        Remove-AppleDouble
        docker build -t zhiji-aml:v0 -f 05_docker\Dockerfile .
        docker run --rm --entrypoint bash zhiji-aml:v0 05_docker/smoke_in_image.sh
    }
    "docker-stop" {
        docker compose -f 05_docker\compose.yml down
    }
    default {
        Write-Error "unknown target: $Target (see .\make.ps1 help)"
    }
}

if ($Target -ne "help") { exit $LASTEXITCODE }
