param(
    [Parameter(Mandatory=$true)]
    [string]$DeepStackURL,

    [Parameter(Mandatory=$false)]
    [string]$APIKEY
)

$ErrorActionPreference = "Stop"

Write-Host "Testing DeepStack : "$DeepStackURL

$env:TEST_DATA_DIR = [IO.Path]::Combine($PSScriptRoot,"test_data")
$env:TEST_DEEPSTACK_URL = $DeepStackURL
$env:TEST_API_KEY = $APIKEY

$p = Start-Process -FilePath "locust" -Wait -NoNewWindow
exit($p.ExitCode)