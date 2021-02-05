param(
    [ValidateSet("CPU","GPU")]
    [Parameter(Mandatory=$true)]
    [string]$Platform,

    [Parameter(Mandatory=$true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

$setup_script = "#include ""environment.iss"""
$setup_script += "`n#define MyAppName ""DeepStack"""
$setup_script += "`n#define MyAppVersion ""$Version"""
$setup_script += "`n#define MyAppPublisher ""DeepQuestAI"""
$setup_script += "`n#define MyAppURL ""https://www.deepstack.cc"""
$setup_script += "`n#define MyAppExeName ""deepstack.exe"""
$setup_script += "`n#define MyAppIcon ""logo.ico"""

$setup_script += "`n`n[Setup]"
$setup_script += "`nChangesEnvironment=true"
$setup_script += "`nAppId={{0E2C3125-3440-4622-A82A-3B1E07310EF2}"
$setup_script += "`nAppName={#MyAppName}"
$setup_script += "`nAppVersion={#MyAppVersion}"
$setup_script += "`nAppPublisher={#MyAppPublisher}"
$setup_script += "`nAppPublisherURL={#MyAppURL}"
$setup_script += "`nAppSupportURL={#MyAppURL}"
$setup_script += "`nAppUpdatesURL={#MyAppURL}"
$setup_script += "`nDefaultDirName=C:\{#MyAppName}"
$setup_script += "`nDisableDirPage=yes"
$setup_script += "`nDefaultGroupName=DeepStack"
$setup_script += "`nOutputBaseFilename=DeepStack-Installer-$Platform"
$setup_script += "`nCompression=lzma"
$setup_script += "`nSolidCompression=yes"

$setup_script += "`n`n[Languages]"
$setup_script += "`nName: ""english""; MessagesFile: ""compiler:Default.isl"""
$setup_script += "`nName: ""armenian""; MessagesFile: ""compiler:Languages\Armenian.isl"""
$setup_script += "`nName: ""brazilianportuguese""; MessagesFile: ""compiler:Languages\BrazilianPortuguese.isl"""
$setup_script += "`nName: ""catalan""; MessagesFile: ""compiler:Languages\Catalan.isl"""
$setup_script += "`nName: ""corsican""; MessagesFile: ""compiler:Languages\Corsican.isl"""
$setup_script += "`nName: ""czech""; MessagesFile: ""compiler:Languages\Czech.isl"""
$setup_script += "`nName: ""danish""; MessagesFile: ""compiler:Languages\Danish.isl"""
$setup_script += "`nName: ""dutch""; MessagesFile: ""compiler:Languages\Dutch.isl"""
$setup_script += "`nName: ""finnish""; MessagesFile: ""compiler:Languages\Finnish.isl"""
$setup_script += "`nName: ""french""; MessagesFile: ""compiler:Languages\French.isl"""
$setup_script += "`nName: ""german""; MessagesFile: ""compiler:Languages\German.isl"""
$setup_script += "`nName: ""hebrew""; MessagesFile: ""compiler:Languages\Hebrew.isl"""
$setup_script += "`nName: ""icelandic""; MessagesFile: ""compiler:Languages\Icelandic.isl"""
$setup_script += "`nName: ""italian""; MessagesFile: ""compiler:Languages\Italian.isl"""
$setup_script += "`nName: ""japanese""; MessagesFile: ""compiler:Languages\Japanese.isl"""
$setup_script += "`nName: ""norwegian""; MessagesFile: ""compiler:Languages\Norwegian.isl"""
$setup_script += "`nName: ""polish""; MessagesFile: ""compiler:Languages\Polish.isl"""
$setup_script += "`nName: ""portuguese""; MessagesFile: ""compiler:Languages\Portuguese.isl"""
$setup_script += "`nName: ""russian""; MessagesFile: ""compiler:Languages\Russian.isl"""
$setup_script += "`nName: ""slovak""; MessagesFile: ""compiler:Languages\Slovak.isl"""
$setup_script += "`nName: ""slovenian""; MessagesFile: ""compiler:Languages\Slovenian.isl"""
$setup_script += "`nName: ""spanish""; MessagesFile: ""compiler:Languages\Spanish.isl"""
$setup_script += "`nName: ""turkish""; MessagesFile: ""compiler:Languages\Turkish.isl"""
$setup_script += "`nName: ""ukrainian""; MessagesFile: ""compiler:Languages\Ukrainian.isl"""


$setup_script += "`n`n[Tasks]"
$setup_script += "`nName: ""desktopicon""; Description: ""{cm:CreateDesktopIcon}""; GroupDescription: ""{cm:AdditionalIcons}""; Flags: unchecked"
$setup_script += "`nName: ""quicklaunchicon""; Description: ""{cm:CreateQuickLaunchIcon}""; GroupDescription: ""{cm:AdditionalIcons}""; Flags: unchecked; OnlyBelowVersion: 0,6.1"

$setup_script += "`n`n[Files]"
$setup_script += "`nSource: ""$PSScriptRoot\server\deepstack.exe""; DestDir: ""{app}""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\*""; DestDir: ""{app}"";"
$setup_script += "`nSource: ""$PSScriptRoot\intelligencelayer\*""; DestDir: ""{app}\intelligencelayer""; Flags: ignoreversion recursesubdirs createallsubdirs"
$setup_script += "`nSource: ""$PSScriptRoot\interpreter\*""; DestDir: ""{app}\interpreter""; Flags: ignoreversion recursesubdirs createallsubdirs"
$setup_script += "`nSource: ""$PSScriptRoot\redis\*""; DestDir: ""{app}\redis""; Flags: ignoreversion recursesubdirs createallsubdirs"
$setup_script += "`nSource: ""$PSScriptRoot\server\*""; DestDir: ""{app}\server""; Flags: ignoreversion recursesubdirs createallsubdirs"
if($Platform -eq "CPU"){
    $setup_script += "`nSource: ""$PSScriptRoot\platform\platform.windows.cpu.json""; DestDir: ""{app}\server""; DestName: ""platform.json""; Flags: ignoreversion"
}
elseif ($Platform -eq "GPU") {
    $setup_script += "`nSource: ""$PSScriptRoot\platform\platform.windows.gpu.json""; DestDir: ""{app}\server""; DestName: ""platform.json""; Flags: ignoreversion"
}
$setup_script += "`nSource: ""$PSScriptRoot\server\version.txt""; DestDir: ""{app}\server""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\sharedfiles\categories_places365.txt""; DestDir: ""{app}\sharedfiles""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\sharedfiles\face.pt""; DestDir: ""{app}\sharedfiles""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\sharedfiles\facerec-high.model""; DestDir: ""{app}\sharedfiles""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\sharedfiles\scene.pt""; DestDir: ""{app}\sharedfiles""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\sharedfiles\yolov5m.pt""; DestDir: ""{app}\sharedfiles""; Flags: ignoreversion"
if($Platform -eq "CPU"){
    $setup_script += "`nSource: ""$PSScriptRoot\windows_packages_cpu\*""; DestDir: ""{app}\windows_packages""; Flags: ignoreversion recursesubdirs createallsubdirs"
} 
elseif($Platform -eq "GPU"){
    $setup_script += "`nSource: ""$PSScriptRoot\windows_packages_gpu\*""; DestDir: ""{app}\windows_packages""; Flags: ignoreversion recursesubdirs createallsubdirs"
}

$setup_script += "`nSource: ""$PSScriptRoot\logo.ico""; DestDir: ""{app}""; Flags: ignoreversion"
$setup_script += "`nSource: ""$PSScriptRoot\init.py""; DestDir: ""{app}""; Flags: ignoreversion"

$setup_script += "`n`n[Icons]"
$setup_script += "`nName: ""{group}\{#MyAppName}""; Filename: ""{app}\{#MyAppExeName}"""
$setup_script += "`nName: ""{group}\{cm:UninstallProgram,{#MyAppName}}""; Filename: ""{uninstallexe}"""
$setup_script += "`nName: ""{commondesktop}\{#MyAppName}""; Filename: ""{app}\{#MyAppExeName}""; IconFilename: {app}\{#MyAppIcon}; Tasks: desktopicon quicklaunchicon"
$setup_script += "`nName: ""{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}""; Filename: ""{app}\{#MyAppExeName}""; IconFilename: {app}\{#MyAppIcon}; Tasks: quicklaunchicon"

$setup_script += "`n[Code]"
$setup_script += "`nprocedure CurStepChanged(CurStep: TSetupStep);"
$setup_script += "`nbegin"
$setup_script += "`n    if CurStep = ssPostInstall"
$setup_script += "`n     then EnvAddPath(ExpandConstant('{app}') + '\server');"
$setup_script += "`nend;"

$setup_script += "`n`nprocedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);"
$setup_script += "`nbegin"
$setup_script += "`n    if CurUninstallStep = usPostUninstall"
$setup_script += "`n    then EnvRemovePath(ExpandConstant('{app}' + '\server'));"
$setup_script += "`nend;"

Set-Content -Path "deepstack-windows.iss" -Value $setup_script