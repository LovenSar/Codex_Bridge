# Stress Test Script for Codex Bridge
param(
    [string]$Url = "http://127.0.0.1:18081/v1/chat/completions",
    [int]$Concurrency = 10,
    [int]$TotalRequests = 100
)

$payload = @{
    model = "Qwen/Qwen3.5-27B"
    messages = @(
        @{role = "user"; content = "Hello"}
    )
    max_tokens = 50
} | ConvertTo-Json

$results = @()
$lock = New-Object System.Threading.Lock

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$jobs = @()
for ($i = 0; $i -lt $TotalRequests; $i++) {
    $jobs += $i
    if ($jobs.Count -ge $Concurrency) {
        # Wait for batch
        Start-Sleep -Milliseconds 100
        $jobs = @()
    }

    $job = Start-Job -ScriptBlock {
        param($u, $p)

        $r = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $resp = Invoke-WebRequest -Uri $u -Method Post -Body $p -ContentType "application/json" -TimeoutSec 60
            $r.Stop()
            @{
                OK = $resp.StatusCode -eq 200
                Status = $resp.StatusCode
                Duration = $r.ElapsedMilliseconds
            }
        } catch {
            $r.Stop()
            @{
                OK = $false
                Status = 0
                Duration = $r.ElapsedMilliseconds
                Error = $_.Exception.Message
            }
        }
    } -ArgumentList $Url, $payload

    if ((Get-Job -State Running).Count -ge $Concurrency) {
        while ((Get-Job -State Running).Count -ge $Concurrency) {
            Start-Sleep -Milliseconds 50
        }
    }
}

# Wait for all jobs
Get-Job | Wait-Job | Out-Null
$results = Get-Job | Receive-Job | ConvertTo-Json
Remove-Job *

$sw.Stop()

$total = $results.Count
$success = ($results | Where-Object { $_.OK }).Count
$failed = $total - $success
$latencies = $results | ForEach-Object { $_.Duration } | Sort-Object

Write-Host ""
Write-Host "=== Stress Test Results ==="
Write-Host "Total Requests: $total"
Write-Host "Success: $success ($(([math]::Round($success/$total*100, 1)))%)"
Write-Host "Failed: $failed"
Write-Host "Total Time: $($sw.Elapsed.TotalSeconds)s"
Write-Host "Min Latency: $($latencies[0])ms"
Write-Host "Max Latency: $($latencies[-1])ms"
Write-Host "Avg Latency: $([math]::Round(($latencies | Measure-Object -Average).Average, 2))ms"