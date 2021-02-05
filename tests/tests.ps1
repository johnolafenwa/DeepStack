param(
    [Parameter(Mandatory=$true)]
    [string]$DeepStackURL,

    [Parameter(Mandatory=$false)]
    [string]$APIKEY
)

$ErrorActionPreference = "Stop"

Write-Host "Testing DeepStack : "$DeepStackURL

$env:TEST_IMAGES_DIR = [IO.Path]::Combine($PSScriptRoot,"test_data")
$env:TEST_DEEPSTACK_URL = $DeepStackURL
$env:TEST_API_KEY = $APIKEY

$python="python3"
if($IsWindows){
    $python = "python"
}

$p = Start-Process -FilePath $python -ArgumentList "-m pytest" -Wait -NoNewWindow
exit($p.ExitCode)