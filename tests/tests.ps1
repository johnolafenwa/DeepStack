param(
    [Parameter(Mandatory=$true)]
    [string]$DeepStackURL,

    [Parameter(Mandatory=$false)]
    [string]$APIKEY
)

Write-Host "Testing DeepStack : "$DeepStackURL

$env:TEST_IMAGES_DIR = [IO.Path]::Combine($PSScriptRoot,"test_images")
$env:TEST_DEEPSTACK_URL = $DeepStackURL
$env:TEST_API_KEY = $APIKEY

Start-Process -FilePath "pytest" -Wait -NoNewWindow
