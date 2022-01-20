function Download($URL, $NAME){

    (New-Object System.Net.WebClient).DownloadFile($URL, $NAME+".zip")
    Expand-Archive -Path $NAME".zip" -Force
    Remove-Item -Path $NAME".zip" -Force

}

Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/sharedfiles.zip" -NAME "sharedfiles"
Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/interpreter.zip" -NAME "interpreter"
Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/redis.zip" -NAME "redis"
Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/gpu_env.zip" -NAME "windows_env_gpu"
Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/cpu_env.zip" -NAME "windows_env_cpu"
Download -URL "https://deepquest.sfo2.digitaloceanspaces.com/deepstack/shared-files/windows_setup.zip" -NAME "windows_setup"
