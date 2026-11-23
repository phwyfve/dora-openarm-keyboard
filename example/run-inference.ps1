$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PATH = "$root\.venv\Scripts;$env:USERPROFILE\.dora\bin;$env:PATH"
$env:PYTHONUNBUFFERED = "1"
$dora = (Get-Command dora.exe -CommandType Application -ErrorAction Stop).Source
Push-Location $PSScriptRoot
try {
    & $dora run dataflow-mujoco-inference.yaml
    if ($LASTEXITCODE -ne 0) { throw "Inference dataflow exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
