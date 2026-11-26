param(
    [string]$Checkpoint,
    [ValidateRange(1,100)][int]$ActionSteps = 100,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$env:PATH = "$root\.venv\Scripts;$env:USERPROFILE\.dora\bin;$env:PATH"
$env:PYTHONUNBUFFERED = "1"
$previousCheckpoint = $env:ACT_POLICY_CHECKPOINT
$previousActionSteps = $env:ACT_ACTION_STEPS
if ($Checkpoint) {
    $resolved = (Resolve-Path -LiteralPath $Checkpoint -ErrorAction Stop).Path
}
Push-Location $PSScriptRoot
try {
    $env:ACT_ACTION_STEPS = [string]$ActionSteps
    Write-Output "ACT execution horizon: $ActionSteps actions before recalculating."
    if ($Checkpoint) {
        foreach ($name in @("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")) {
            if (-not (Test-Path -LiteralPath (Join-Path $resolved $name) -PathType Leaf)) {
                throw "Incomplete local ACT checkpoint: $resolved\$name"
            }
        }
        $env:ACT_POLICY_CHECKPOINT = $resolved
        Write-Output "Testing local ACT checkpoint: $resolved"
    } else {
        Remove-Item Env:\ACT_POLICY_CHECKPOINT -ErrorAction SilentlyContinue
        Write-Output "Testing the original pinned pretrained ACT."
    }
    if ($Check) {
        & "$root\.venv-inference\Scripts\python.exe" -u act-inference.py --check
        if ($LASTEXITCODE -ne 0) { throw "ACT checkpoint check exited with code $LASTEXITCODE" }
    } else {
        $dora = (Get-Command dora.exe -CommandType Application -ErrorAction Stop).Source
        & $dora run dataflow-mujoco-inference.yaml
        if ($LASTEXITCODE -ne 0) { throw "Inference dataflow exited with code $LASTEXITCODE" }
    }
} finally {
    if ($null -eq $previousActionSteps) {
        Remove-Item Env:\ACT_ACTION_STEPS -ErrorAction SilentlyContinue
    } else {
        $env:ACT_ACTION_STEPS = $previousActionSteps
    }
    if ($null -eq $previousCheckpoint) {
        Remove-Item Env:\ACT_POLICY_CHECKPOINT -ErrorAction SilentlyContinue
    } else {
        $env:ACT_POLICY_CHECKPOINT = $previousCheckpoint
    }
    Pop-Location
}
