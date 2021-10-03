function Download($URL, $NAME){

    (New-Object System.Net.WebClient).DownloadFile($URL, $NAME+".zip")
    Expand-Archive -Path $NAME".zip" -Force
    Remove-Item -Path $NAME".zip" -Force

}

Download -URL "https://deepstack.blob.core.windows.net/shared-files/sharedfiles.zip" -NAME "sharedfiles"
Download -URL "https://deepstack.blob.core.windows.net/shared-files/interpreter.zip" -NAME "interpreter"
Download -URL "https://deepstack.blob.core.windows.net/shared-files/redis.zip" -NAME "redis"
Download -URL "https://deepstack.blob.core.windows.net/shared-files/windows_packages_cpu.zip" -NAME "windows_packages_cpu"
Download -URL "https://deepstack.blob.core.windows.net/shared-files/windows_packages_gpu.zip" -NAME "windows_packages_gpu"
Download -URL "https://deepstack.blob.core.windows.net/shared-files/windows_setup.zip" -NAME "windows_setup"
