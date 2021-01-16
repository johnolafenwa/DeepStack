Write-Host "Setting Env Variables"

$DATA_DIR = "datastore"
$TEMP_PATH = "deeptemp"

if (!(Test-Path $DATA_DIR -PathType Container)){
    mkdir $DATA_DIR
}

if (!(Test-Path $TEMP_PATH -PathType Container)){
    mkdir $TEMP_PATH
}

$env:DATA_DIR = [IO.Path]::Combine($PSScriptRoot,"datastore")
$env:TEMP_PATH = [IO.Path]::Combine($PSScriptRoot,"deeptemp")
$env:APPDIR = $PSScriptRoot
$env:PROFILE = "desktop_cpu"
$env:SLEEP_TIME = "0.01"
$env:CUDA_MODE = "False"


