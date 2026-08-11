#!/usr/bin/env pwsh
# Staging-only experiment. NOT part of the repo test suite.
#
# Question: does the single-AMD-GPU miss in #8335 actually reproduce under Windows PowerShell
# 5.1, and is it fixed by the @() wrap in unslothai/unsloth#8398?
#
# The claim under test is: `$wmiGpus = if (...) { $healthyGpus } else { $amdGpus }` lets a
# one-element array unroll to a scalar, and a scalar's .Count answers $null on 5.1, so the
# `-gt 0` guard below it is false and the host reports no AMD GPU at all.
#
# That claim is NOT self-evident. PowerShell v3 gave scalars an intrinsic .Count of 1, and the
# documented exception is [pscustomobject] (fixed in 6.1), not every scalar. A CimInstance is a
# plain .NET object, so it should GET the intrinsic 1 even on 5.1 -- in which case the stated
# mechanism is wrong. So this script measures .Count per object flavour instead of assuming,
# and runs the real pre-fix and post-fix source regions lifted straight out of setup.ps1.
#
#   pre-fix  region  -> from upstream main's studio/setup.ps1
#   post-fix region  -> from the PR head's studio/setup.ps1
#
# Usage: pwsh|powershell -NoProfile -File ps51_wmi_unroll_proof.ps1 -PreFix <path> -PostFix <path>

param(
    [Parameter(Mandatory = $true)][string]$PreFix,
    [Parameter(Mandatory = $true)][string]$PostFix
)

$ErrorActionPreference = "Stop"

$isDesktop = ($PSVersionTable.PSVersion.Major -lt 6)
$tag = if ($isDesktop) { "5.1" } else { "7" }
Write-Host "================================================================"
Write-Host "interpreter          : $($PSVersionTable.PSVersion)"
Write-Host "PSEdition            : $($PSVersionTable.PSEdition)"
Write-Host "treated as           : Windows PowerShell $tag"
Write-Host "OS                   : $([System.Environment]::OSVersion.VersionString)"
Write-Host "================================================================"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# ---------------------------------------------------------------------------------------------
# Lift the region out of both files rather than retyping it, so this cannot drift from setup.ps1.
# Ends on the closing brace of the `if ($wmiGpus.Count -gt 0)` block, not on the last statement
# inside it, or the lifted text is not parseable on its own.
$regionPat = '(?s)(\$amdGpus = @\(Get-CimInstance Win32_VideoController.*?\$ROCmGpuLabel = \$script:ROCmGpuLabels\[0\]\n\s*\})'

function Get-Region([string]$path) {
    $text = (Get-Content -Raw -LiteralPath $path) -replace "`r`n", "`n"
    if ($text -match $regionPat) { return $Matches[1] }
    throw "could not lift the AMD WMI region out of $path"
}

$preRegion = Get-Region $PreFix
$postRegion = Get-Region $PostFix

Write-Host ""
Write-Host "=== the two source shapes under test ==="
Check "pre-fix source has the UNWRAPPED if-expression" (
    $preRegion -match '\$wmiGpus = if \(\$healthyGpus\.Count -gt 0\)')
Check "pre-fix source does NOT have the @() wrap" (
    -not ($preRegion -match '\$wmiGpus = @\(if'))
Check "post-fix source HAS the @() wrap" (
    $postRegion -match '\$wmiGpus = @\(if \(\$healthyGpus\.Count -gt 0\) \{ \$healthyGpus \} else \{ \$amdGpus \}\)')
Check "post-fix source does NOT still have the unwrapped form" (
    -not ($postRegion -match '(?m)^\s*\$wmiGpus = if \('))
if ($failures -gt 0) {
    Write-Host ""
    Write-Host "ABORT: the pre-fix file does not look pre-fix (has main already got the wrap?)." -ForegroundColor Red
    exit 2
}
Write-Host "  pre  : $((($preRegion -split "`n") | Where-Object { $_ -match '\$wmiGpus =' }).Trim())"
Write-Host "  post : $((($postRegion -split "`n") | Where-Object { $_ -match '\$wmiGpus =' }).Trim())"

# ---------------------------------------------------------------------------------------------
# Object flavours. The type matters: the intrinsic .Count is added to scalars that do not
# already carry one, and [pscustomobject] is the documented 5.1 hole. Get-CimInstance returns
# CimInstance, so THAT is the flavour the claim actually stands or falls on. New-CimInstance
# -ClientOnly builds the same .NET type without needing a real Radeon in the box.
if (-not ("StagingFakeVideoController" -as [type])) {
    # A plain .NET object with no Count of its own -- the same shape of thing CimInstance is,
    # and constructible on a Linux pwsh where the CIM cmdlets do not exist. Lets the script be
    # dry-run off Windows; the CimInstance flavour is the one that decides the verdict.
    Add-Type -TypeDefinition @"
public class StagingFakeVideoController {
    public string Name;
    public object ConfigManagerErrorCode;
}
"@
}

$HasNewCimInstance = [bool](Get-Command New-CimInstance -ErrorAction SilentlyContinue)
$Flavours = @()
if ($HasNewCimInstance) { $Flavours += "ciminstance" }
$Flavours += @("dotnetobject", "pscustomobject", "selected")

function New-Gpu([string]$Name, $ErrCode, [string]$Flavour) {
    switch ($Flavour) {
        "dotnetobject" {
            $o = New-Object StagingFakeVideoController
            $o.Name = $Name
            $o.ConfigManagerErrorCode = $(if ($null -ne $ErrCode) { [uint32]$ErrCode } else { $null })
            return $o
        }
        "ciminstance" {
            $props = @{ Name = $Name }
            if ($null -ne $ErrCode) { $props["ConfigManagerErrorCode"] = [uint32]$ErrCode }
            return (New-CimInstance -ClassName Win32_VideoController -ClientOnly -Property $props)
        }
        "pscustomobject" {
            if ($null -ne $ErrCode) { return [pscustomobject]@{ Name = $Name; ConfigManagerErrorCode = [uint32]$ErrCode } }
            return [pscustomobject]@{ Name = $Name }
        }
        "selected" {
            # What `| Select-Object` yields: Selected.* is a PSCustomObject, the flavour that
            # really does answer $null to .Count on 5.1.
            $o = [pscustomobject]@{ Name = $Name; ConfigManagerErrorCode = $(if ($null -ne $ErrCode) { [uint32]$ErrCode } else { $null }) }
            return ($o | Select-Object Name, ConfigManagerErrorCode)
        }
        default { throw "unknown flavour $Flavour" }
    }
}

Write-Host ""
Write-Host "=== raw scalar .Count per object flavour (the crux) ==="
# This is the measurement the whole diagnosis rests on. Printed, not just asserted, so the
# answer is legible even when a check below is red.
$countReport = @{}
foreach ($flavour in $Flavours) {
    $one = @((New-Gpu "AMD Radeon RX 7900 XTX" 0 $flavour))
    # The unroll itself: a one-element array leaving an if-expression.
    $scalar = if ($one.Count -gt 0) { $one } else { @() }
    $isArr = $scalar -is [array]
    $c = $scalar.Count
    $countReport[$flavour] = $c
    $shown = if ($null -eq $c) { '$null' } else { "$c" }
    Write-Host ("  {0,-16} type={1,-46} unrolled={2,-5} .Count={3}" -f `
        $flavour, $scalar.GetType().FullName, (-not $isArr), $shown)
    Check "$flavour : the one-element array unrolls out of the if-expression" (-not $isArr)
}

# The runner's own adapter, as an untouched real WMI object. No filtering, no construction --
# purely a datapoint on what a genuine Get-CimInstance scalar answers on this interpreter.
if ($IsWindows -or $isDesktop) {
    try {
        $real = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)
        Write-Host "  runner adapters  : $($real.Count) -> $(($real | ForEach-Object { $_.Name }) -join ', ')"
        if ($real.Count -ge 1) {
            $realScalar = if ($real.Count -gt 0) { @($real[0]) | ForEach-Object { $_ } } else { $null }
            $rc = $real[0].Count
            Write-Host "  real CimInstance scalar .Count = $(if ($null -eq $rc) { '$null' } else { $rc })"
        }
    } catch { Write-Host "  (real WMI read failed: $($_.Exception.Message))" }
}

# ---------------------------------------------------------------------------------------------
# Run the lifted region for real, with Get-CimInstance stubbed.
function Invoke-Region {
    param([string]$Region, $Gpus)
    $sb = [scriptblock]::Create(@"
param(`$Gpus)
function Get-CimInstance {
    [CmdletBinding()]
    param([Parameter(Position = 0)] `$ClassName,
          [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    return `$Gpus
}
`$script:ROCmGpuLabels = `$null
`$ROCmGpuLabel = `$null
$Region
`$n = if (`$null -eq `$script:ROCmGpuLabels) { 0 } else { @(`$script:ROCmGpuLabels).Count }
[pscustomobject]@{ Detected = `$n; Label = `$ROCmGpuLabel }
"@)
    return (& $sb $Gpus)
}

# Every scenario the WMI branch can meet on a real host. `Expect` is what a correct
# implementation must report, and it is the same number on 5.1 and on 7.
function Scenarios([string]$Flavour) {
    $healthy = { New-Gpu "AMD Radeon RX 7900 XTX" 0 $Flavour }
    return @(
        @{ Name = "one healthy AMD GPU";           Gpus = @((& $healthy));                                             Expect = 1 },
        @{ Name = "two healthy AMD GPUs";          Gpus = @((New-Gpu "AMD Radeon 890M" 0 $Flavour), (New-Gpu "AMD Radeon RX 7900 XTX" 0 $Flavour)); Expect = 2 },
        @{ Name = "one AMD GPU, error code 45";    Gpus = @((New-Gpu "AMD Radeon RX 7900 XTX" 45 $Flavour));           Expect = 1 },
        @{ Name = "one healthy + one code 45";     Gpus = @((New-Gpu "AMD Radeon 890M" 0 $Flavour), (New-Gpu "AMD Radeon RX 7900 XTX" 45 $Flavour)); Expect = 1 },
        @{ Name = "AMD GPU with no error code";    Gpus = @((New-Gpu "AMD Radeon 8060S Graphics" $null $Flavour));     Expect = 1 },
        @{ Name = "one healthy AMD + one NVIDIA";  Gpus = @((New-Gpu "NVIDIA GeForce RTX 4090" 0 $Flavour), (New-Gpu "AMD Radeon 890M" 0 $Flavour)); Expect = 1 },
        @{ Name = "no GPUs at all";                Gpus = @();                                                          Expect = 0 },
        @{ Name = "WMI returned null";             Gpus = $null;                                                        Expect = 0 },
        @{ Name = "NVIDIA only";                   Gpus = @((New-Gpu "NVIDIA GeForce RTX 4090" 0 $Flavour));            Expect = 0 }
    )
}

$divergence = @{}
foreach ($flavour in $Flavours) {
    Write-Host ""
    Write-Host "=== scenario sweep, flavour = $flavour ==="
    Write-Host ("  {0,-30} {1,6} {2,8} {3,9}" -f "scenario", "expect", "pre-fix", "post-fix")
    $diverged = @()
    foreach ($s in (Scenarios $flavour)) {
        $pre = (Invoke-Region $preRegion $s.Gpus).Detected
        $post = (Invoke-Region $postRegion $s.Gpus).Detected
        Write-Host ("  {0,-30} {1,6} {2,8} {3,9}" -f $s.Name, $s.Expect, $pre, $post)
        # The post-fix form is a hard gate: it must be right on every scenario, on BOTH
        # interpreters. A regression here fails the job.
        Check "post-fix, $flavour : $($s.Name) -> $($s.Expect)" ($post -eq $s.Expect)
        if ($pre -ne $s.Expect) { $diverged += $s.Name }
    }
    $divergence[$flavour] = $diverged
    if ($diverged.Count -gt 0) {
        Write-Host "  pre-fix is WRONG on: $($diverged -join '; ')" -ForegroundColor Yellow
    } else {
        Write-Host "  pre-fix agreed with post-fix on every scenario on this interpreter."
    }
}

# ---------------------------------------------------------------------------------------------
Write-Host ""
Write-Host "=== VERDICT ($tag) ==="
$cimKey = if ($HasNewCimInstance) { "ciminstance" } else { "dotnetobject" }
$cim = $divergence[$cimKey]
if ($isDesktop) {
    Write-Host "Windows PowerShell 5.1. This is the interpreter #8335 was reported on."
    if ($cim.Count -gt 0) {
        Write-Host "CONFIRMED: with real $cimKey objects the pre-fix source misses the GPU on:" -ForegroundColor Green
        $cim | ForEach-Object { Write-Host "    - $_" -ForegroundColor Green }
        Write-Host "The @() wrap fixes every one of them. The #8335 diagnosis holds." -ForegroundColor Green
    } else {
        Write-Host "NOT REPRODUCED with CimInstance objects on 5.1." -ForegroundColor Red
        Write-Host "A scalar CimInstance answered .Count = $($countReport['ciminstance'])," -ForegroundColor Red
        Write-Host "so the stated mechanism (a scalar's .Count is `$null on 5.1) does NOT hold" -ForegroundColor Red
        Write-Host "for the objects Get-CimInstance actually returns. The @() wrap is still the" -ForegroundColor Red
        Write-Host "right idiom, but the root cause of #8335 is something else and needs a" -ForegroundColor Red
        Write-Host "second look. pscustomobject divergence: $($divergence['pscustomobject'] -join '; ')" -ForegroundColor Red
        Write-Host "                selected divergence: $($divergence['selected'] -join '; ')" -ForegroundColor Red
        # Loud, but not a job failure: this is a finding to report, and failing here would
        # hide the post-fix gate results underneath it.
    }
} else {
    Write-Host "PowerShell 7. Recorded to show what PS7-only testing can and cannot see."
    if ($cim.Count -eq 0) {
        Write-Host "As expected: pre-fix and post-fix agree everywhere here, so a pwsh-only" -ForegroundColor Yellow
        Write-Host "test suite is blind to this class of bug." -ForegroundColor Yellow
    } else {
        Write-Host "UNEXPECTED: the pre-fix form is wrong on PowerShell 7 too, on: $($cim -join '; ')" -ForegroundColor Yellow
    }
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All post-fix checks passed" -ForegroundColor Green
