call C:\DeepStack\windows_env\Scripts\Activate.bat

if "%2"=="" ( python %1 ) else ( python %1 --model %2 --name %3 )
