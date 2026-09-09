# Deterministic structural comparison of the installer scripts, base against head.
#
# A textual diff says what changed. This says what changed STRUCTURALLY: which named
# functions exist on each side, which were added, removed or altered, and which call sites
# moved. The point is to prove the change is confined to the functions it claims, rather
# than trusting a reviewer's eye over 2000 diff lines.
#
# Uses the PowerShell language parser, so a comment-only or whitespace-only edit does not
# register as a change, and a real edit cannot hide inside one.

param(
    [Parameter(Mandatory = $true)][string]$Base,
    [Parameter(Mandatory = $true)][string]$Head
)

function Get-Functions {
    param([string]$Path)
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$errors)
    if ($errors -and $errors.Count) {
        Write-Host "PARSE ERRORS in $Path : $($errors.Count)"
        foreach ($e in $errors[0..([Math]::Min(4, $errors.Count - 1))]) { Write-Host "   $($e.Message)" }
    }
    $found = @{}
    foreach ($fn in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
        # Body text with comments and blank lines stripped, so cosmetic edits do not count
        # as behaviour changes.
        $body = $fn.Extent.Text -split "`n" |
            ForEach-Object { ($_ -replace '(?<!`)#.*$', '').TrimEnd() } |
            Where-Object { $_.Trim() -ne '' }
        $found[$fn.Name] = ($body -join "`n")
    }
    return $found
}

function Get-CommandNames {
    param([string]$Path)
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$null)
    $names = @{}
    foreach ($cmd in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.CommandAst] }, $true)) {
        $n = $cmd.GetCommandName()
        if ($n) { $names[$n] = 1 + ($names[$n] | ForEach-Object { $_ }) }
    }
    return $names
}

$b = Get-Functions -Path $Base
$h = Get-Functions -Path $Head

$added   = @($h.Keys | Where-Object { -not $b.ContainsKey($_) } | Sort-Object)
$removed = @($b.Keys | Where-Object { -not $h.ContainsKey($_) } | Sort-Object)
$changed = @($h.Keys | Where-Object { $b.ContainsKey($_) -and $b[$_] -ne $h[$_] } | Sort-Object)
$same    = @($h.Keys | Where-Object { $b.ContainsKey($_) -and $b[$_] -eq $h[$_] })

Write-Host "=== $(Split-Path $Head -Leaf) ==="
Write-Host "  functions: base $($b.Count), head $($h.Count), identical $($same.Count)"
Write-Host "  ADDED   ($($added.Count)): $($added -join ', ')"
Write-Host "  REMOVED ($($removed.Count)): $($removed -join ', ')"
Write-Host "  CHANGED ($($changed.Count)): $($changed -join ', ')"

$bc = Get-CommandNames -Path $Base
$hc = Get-CommandNames -Path $Head
foreach ($watch in @('Add-Type', 'Invoke-Expression', 'iex', 'Start-Process', 'New-Object')) {
    $bn = if ($bc.ContainsKey($watch)) { $bc[$watch] } else { 0 }
    $hn = if ($hc.ContainsKey($watch)) { $hc[$watch] } else { 0 }
    $flag = if ($bn -ne $hn) { "  <-- CHANGED" } else { "" }
    Write-Host "  command '$watch': base $bn, head $hn$flag"
}
Write-Host ""
