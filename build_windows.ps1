$ErrorActionPreference = "Stop"
Start-Process -FilePath "windows_setup/ISCC.exe" -ArgumentList "deepstack-windows.iss" -Wait -NoNewWindow