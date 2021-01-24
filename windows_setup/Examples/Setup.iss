; -- Setup.iss --
; Inno Setup QuickStart Pack's own Setup script

; Inno Setup
; Copyright (C) 1997-2020 Jordan Russell. All rights reserved.
; Portions Copyright (C) 2000-2020 Martijn Laan. All rights reserved.
; For conditions of distribution and use, see LICENSE.TXT.

#include "isdonateandmail.iss"

#include "isportable.iss"

[Setup]
AppName=Inno Setup QuickStart Pack
AppId={code:GetAppId|Inno Setup 6}
AppVersion=6.1.2
AppPublisher=jrsoftware.org
AppPublisherURL=https://www.innosetup.com/
AppSupportURL=https://www.innosetup.com/
AppUpdatesURL=https://www.innosetup.com/
VersionInfoCopyright=Copyright (C) 1997-2020 Jordan Russell. Portions Copyright (C) 2000-2020 Martijn Laan.
AppMutex=InnoSetupCompilerAppMutex,Global\InnoSetupCompilerAppMutex
SetupMutex=InnoSetupCompilerSetupMutex,Global\InnoSetupCompilerSetupMutex
WizardStyle=modern
DefaultDirName={code:GetDefaultDirName|Inno Setup 6}
DefaultGroupName=Inno Setup 6
PrivilegesRequiredOverridesAllowed=commandline
AllowNoIcons=yes
Compression=lzma2/ultra
InternalCompressLevel=ultra
SolidCompression=yes
Uninstallable=not PortableCheck
UninstallDisplayIcon={app}\Compil32.exe
UsePreviousLanguage=no
LicenseFile=isfiles\license.txt
AppModifyPath="{app}\Ispack-setup.exe" /modify=1
WizardImageFile=compiler:WizModernImage-IS.bmp
WizardSmallImageFile=compiler:WizModernSmallImage-IS.bmp
SetupIconFile=Setup.ico
#ifdef SIGNTOOL
SignTool=ispacksigntool
SignTool=ispacksigntool256
SignedUninstaller=yes
#endif

[Tasks]
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; Check: not PortableCheck
;Name: fileassoc; Description: "{cm:AssocFileExtension,Inno Setup,.iss}"

[InstallDelete]
; Remove old ISPP files
Type: files; Name: {app}\ISCmplr.dls
Type: files; Name: {app}\Builtins.iss
; Remove optional ISCrypt files
Type: files; Name: {app}\ISCrypt.dll
; Remove desktop icon if needed
Type: files; Name: {autodesktop}\Inno Setup Compiler.lnk; Tasks: not desktopicon; Check: not PortableCheck
; Remove old FAQ file
Type: files; Name: "{app}\isfaq.htm"
; Remove old .islu files
Type: files; Name: "{app}\Languages\*.islu"
; Remove translations in case any got demoted
Type: files; Name: "{app}\Languages\*.isl"
; Remove old example files
Type: files; Name: "{app}\Examples\Donate.iss"
Type: files; Name: "{app}\Examples\Donate.bmp"

[Files]
; First the files used by [Code] so these can be quickly decompressed despite solid compression
Source: "otherfiles\IDE.ico"; Flags: dontcopy
Source: "otherfiles\ISCrypt.ico"; Flags: dontcopy
; Other files
Source: "isfiles\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "isfiles\Examples\*"; DestDir: "{app}\Examples"; Flags: recursesubdirs ignoreversion
Source: "Setup.iss"; DestDir: "{app}\Examples"; Flags: ignoreversion
Source: "Setup.ico"; DestDir: "{app}\Examples"; Flags: ignoreversion
Source: "IsDonateAndMail.iss"; DestDir: "{app}\Examples"; Flags: ignoreversion
Source: "IsDonate.bmp"; DestDir: "{app}\Examples"; Flags: ignoreversion
Source: "IsMail.bmp"; DestDir: "{app}\Examples"; Flags: ignoreversion
Source: "IsPortable.iss"; DestDir: "{app}\Examples"; Flags: ignoreversion
; External files
Source: "{tmp}\ISCrypt.dll"; DestDir: "{app}"; Flags: external ignoreversion; Check: ISCryptCheck
Source: "{srcexe}"; DestDir: "{app}"; DestName: "Ispack-setup.exe"; Flags: external ignoreversion; Check: not ModifyingCheck

[UninstallDelete]
Type: files; Name: "{app}\isfaq.url"
Type: files; Name: "{app}\Examples\Output\setup.exe"
Type: files; Name: "{app}\Examples\Output\setup-*.bin"
Type: dirifempty; Name: "{app}\Examples\Output"
Type: dirifempty; Name: "{app}\Examples\MyDll\Delphi"
Type: dirifempty; Name: "{app}\Examples\MyDll\C#"
Type: dirifempty; Name: "{app}\Examples\MyDll\C"
Type: dirifempty; Name: "{app}\Examples\MyDll"
Type: dirifempty; Name: "{app}\Examples"

[INI]
Filename: "{app}\isfaq.url"; Section: "InternetShortcut"; Key: "URL"; String: "https://jrsoftware.org/isfaq.php" 

[Icons]
; All these will be automatically skipped on portable mode, either because of NoIconsCheck being checked, or because of the desktopicon task being removed
Name: "{group}\Inno Setup Compiler"; Filename: "{app}\Compil32.exe"; WorkingDir: "{app}"; AppUserModelID: "JR.InnoSetup.IDE.6"
Name: "{group}\Inno Setup Documentation"; Filename: "{app}\ISetup.chm";
Name: "{group}\Inno Setup Example Scripts"; Filename: "{app}\Examples\";
Name: "{group}\Inno Setup Preprocessor Documentation"; Filename: "{app}\ISPP.chm";
Name: "{group}\Inno Setup FAQ"; Filename: "{app}\isfaq.url";
Name: "{group}\Inno Setup Revision History"; Filename: "{app}\whatsnew.htm";
Name: "{autodesktop}\Inno Setup Compiler"; Filename: "{app}\Compil32.exe"; WorkingDir: "{app}"; AppUserModelID: "JR.InnoSetup.IDE.6"; Tasks: desktopicon; Check: not AnyIDECheck

[Run]
Filename: "{tmp}\innoide-setup.exe"; StatusMsg: "Installing InnoIDE..."; Parameters: "/verysilent /group=""{groupname}\InnoIDE"" /mergetasks=""desktopicon,file_association"""; Flags: skipifdoesntexist; Check: InnoIDECheck; Tasks: desktopicon
Filename: "{tmp}\innoide-setup.exe"; StatusMsg: "Installing InnoIDE..."; Parameters: "/verysilent /group=""{groupname}\InnoIDE"" /mergetasks=""!desktopicon,file_association"""; Flags: skipifdoesntexist; Check: InnoIDECheck; Tasks: not desktopicon
Filename: "{tmp}\isstudio-setup.exe"; StatusMsg: "Installing Inno Script Studio..."; Parameters: {code:GetISStudioCmdLine}; Flags: skipifdoesntexist; Check: ISStudioCheck
Filename: "{app}\Compil32.exe"; Parameters: "/ASSOC"; StatusMsg: "{cm:AssocingFileExtension,Inno Setup,.iss}"; Check: not PortableCheck and not AnyIDECheck
Filename: "{app}\Compil32.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Inno Setup}"; Flags: nowait postinstall skipifsilent; Check: not AnyIDECheck and not ModifyingCheck
Filename: "{code:GetInnoIDEPath}\InnoIDE.exe"; WorkingDir: "{code:GetInnoIDEPath}"; Description: "{cm:LaunchProgram,InnoIDE}"; Flags: nowait postinstall skipifsilent skipifdoesntexist; Check: InnoIDECheck and not ModifyingCheck
Filename: "{code:GetISStudioPath}\ISStudio.exe"; WorkingDir: "{code:GetISStudioPath}"; Description: "{cm:LaunchProgram,Inno Script Studio}"; Flags: nowait postinstall skipifsilent skipifdoesntexist; Check: ISStudioCheck and not ModifyingCheck

[UninstallRun]
; The /UNASSOC line will be automatically skipped on portable mode, because of Uninstallable begin set to no
Filename: "{app}\Compil32.exe"; Parameters: "/UNASSOC"; RunOnceId: "RemoveISSAssoc"

[Code]
var
  Modifying, AllowInnoIDE: Boolean;

  IDEPage, ISCryptPage: TWizardPage;
  InnoIDECheckBox, ISStudioCheckBox, ISCryptCheckBox: TCheckBox;
  IDEOrg: Boolean;

  DownloadPage: TDownloadWizardPage;
  FilesDownloaded: Boolean;
  
  InnoIDEPath, ISStudioPath: String;
  InnoIDEPathRead, ISStudioPathRead: Boolean;

function GetModuleHandle(lpModuleName: LongInt): LongInt;
external 'GetModuleHandleA@kernel32.dll stdcall';
function ExtractIcon(hInst: LongInt; lpszExeFileName: String; nIconIndex: LongInt): LongInt;
external 'ExtractIconW@shell32.dll stdcall';
function DrawIconEx(hdc: LongInt; xLeft, yTop: Integer; hIcon: LongInt; cxWidth, cyWidth: Integer; istepIfAniCur: LongInt; hbrFlickerFreeDraw, diFlags: LongInt): LongInt;
external 'DrawIconEx@user32.dll stdcall';
function DestroyIcon(hIcon: LongInt): LongInt;
external 'DestroyIcon@user32.dll stdcall';

const
  DI_NORMAL = 3;
  
procedure SetInnoIDECheckBoxChecked(Checked: Boolean);
begin
  if InnoIDECheckBox <> nil then
    InnoIDECheckBox.Checked := Checked;
end;

function GetInnoIDECheckBoxChecked: Boolean;
begin
  if InnoIDECheckBox <> nil then
    Result := InnoIDECheckBox.Checked
  else
    Result := False;
end;

function InitializeSetup(): Boolean;
begin
  Modifying := ExpandConstant('{param:modify|0}') = '1';
  AllowInnoIDE := ExpandConstant('{param:allowinnoide|0}') = '1';
  FilesDownloaded := False;
  InnoIDEPathRead := False;
  ISStudioPathRead := False;
    
  Result := True;
end;

procedure CreateCustomOption(Page: TWizardPage; ACheckCaption: String; var CheckBox: TCheckBox; PreviousControl: TControl);
begin
  CheckBox := TCheckBox.Create(Page);
  with CheckBox do begin
    Top := PreviousControl.Top + PreviousControl.Height + ScaleY(12);
    Width := Page.SurfaceWidth;
    Height := ScaleY(Height);
    Anchors := [akLeft, akTop, akRight];
    Caption := ACheckCaption;
    Parent := Page.Surface;
  end;
end;

function CreateCustomOptionPage(AAfterId: Integer; ACaption, ASubCaption, AIconFileName, ALabel1Caption, ALabel2Caption,
  ACheckCaption: String; var CheckBox: TCheckBox): TWizardPage;
var
  Page: TWizardPage;
  Rect: TRect;
  hIcon: LongInt;
  Label1, Label2: TNewStaticText;
begin
  Page := CreateCustomPage(AAfterID, ACaption, ASubCaption);
  
  try
    AIconFileName := ExpandConstant('{tmp}\' + AIconFileName);
    if not FileExists(AIconFileName) then
      ExtractTemporaryFile(ExtractFileName(AIconFileName));

    Rect.Left := 0;
    Rect.Top := 0;
    Rect.Right := 32;
    Rect.Bottom := 32;

    hIcon := ExtractIcon(GetModuleHandle(0), AIconFileName, 0);
    try
      with TBitmapImage.Create(Page) do begin
        with Bitmap do begin
          Width := 32;
          Height := 32;
          Canvas.Brush.Color := Page.SurfaceColor;
          Canvas.FillRect(Rect);
          DrawIconEx(Canvas.Handle, 0, 0, hIcon, 32, 32, 0, 0, DI_NORMAL);
        end;
        Width := Bitmap.Width;
        Height := Bitmap.Width;
        Parent := Page.Surface;
      end;
    finally
      DestroyIcon(hIcon);
    end;
  except
  end;

  Label1 := TNewStaticText.Create(Page);
  with Label1 do begin
    AutoSize := False;
    Left := WizardForm.SelectDirLabel.Left;
    Width := Page.SurfaceWidth - Left;
    Anchors := [akLeft, akTop, akRight];
    WordWrap := True;
    Caption := ALabel1Caption;
    Parent := Page.Surface;
  end;
  WizardForm.AdjustLabelHeight(Label1);

  Label2 := TNewStaticText.Create(Page);
  with Label2 do begin
    Top := Label1.Top + Label1.Height + ScaleY(12);
    Width := Page.SurfaceWidth;
    Anchors := [akLeft, akTop, akRight];
    WordWrap := True;
    Caption := ALabel2Caption;
    Parent := Page.Surface;
  end;
  WizardForm.AdjustLabelHeight(Label2);
  
  CreateCustomOption(Page, ACheckCaption, CheckBox, Label2);

  Result := Page;
end;

procedure URLLabelOnClick(Sender: TObject);
var
  ErrorCode: Integer;
begin
  ShellExecAsOriginalUser('open', TNewStaticText(Sender).Caption, '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
end;

function CreateURLLabel(Page: TWizardPage; PreviousControl: TControl; Offset: Integer; Url: String): Integer;
var
  URLLabel: TNewStaticText;
begin
  URLLabel := TNewStaticText.Create(Page);
  with URLLabel do begin
    Top := PreviousControl.Top + PreviousControl.Height + ScaleY(12);
    Left := Offset;
    Caption := Url;
    Cursor := crHand;
    OnClick := @UrlLabelOnClick;
    Parent := Page.Surface;
    { Alter Font *after* setting Parent so the correct defaults are inherited first }
    URLLabel.Font.Style := URLLabel.Font.Style + [fsUnderline];
    URLLabel.Font.Color := clBlue;
  end;
  WizardForm.AdjustLabelHeight(URLLabel);
  Result := URLLabel.Width;
end;

function GetIDECheckCaption(const Caption: String): String;
begin
  Result := Caption;
  if PortableCheck then
    Result := Result + ' (Not portable!)';
end;

procedure CreateCustomPages;
var
  Caption, SubCaption1, IconFileName, Label1Caption, Label2Caption, CheckCaption: String;
  UrlSize: Integer;
begin
  if AllowInnoIDE then begin
    Caption := 'InnoIDE and Inno Script Studio';
    SubCaption1 := 'Would you like to download and install InnoIDE or Inno Script Studio?';
    IconFileName := 'IDE.ico';
    Label1Caption :=
      'InnoIDE and Inno Script Studio are easy to use Inno Setup Script editors meant as a replacement of the standard Compiler IDE that comes with Inno Setup.' +
      ' InnoIDE is by Graham Murt, see http://www.innoide.org/ for more information.' +
      ' Inno Script Studio is by Kymoto Solutions, see https://www.kymoto.org/inno-script-studio for more information.'  +  #13#10#13#10 +
      'Using InnoIDE or Inno Script Studio is especially recommended for new users.';
    Label2Caption := 'Select whether you would like to download and install InnoIDE or Inno Script Studio, then click Next.';
    CheckCaption := GetIDECheckCaption('&Download and install InnoIDE');

    IDEPage := CreateCustomOptionPage(wpSelectProgramGroup, Caption, SubCaption1, IconFileName, Label1Caption, Label2Caption, CheckCaption, InnoIDECheckBox);

    CheckCaption := GetIDECheckCaption('D&ownload and install Inno Script Studio');
    CreateCustomOption(IDEPage, CheckCaption, ISStudioCheckBox, InnoIDECheckBox);

    UrlSize := CreateUrlLabel(IDEPage, ISStudioCheckBox, 0, 'http://www.innoide.org/');    
    CreateUrlLabel(IDEPage, ISStudioCheckBox, UrlSize + ScaleX(12), 'https://www.kymoto.org/inno-script-studio');    
  end else begin
    Caption := 'Inno Script Studio';
    SubCaption1 := 'Would you like to download and install Inno Script Studio?';
    IconFileName := 'IDE.ico';
    Label1Caption :=
      'Inno Script Studio is an easy to use Inno Setup Script editor meant as a replacement of the standard Compiler IDE that comes with Inno Setup.' +
      ' Inno Script Studio is by Kymoto Solutions, see https://www.kymoto.org/inno-script-studio for more information.'  +  #13#10#13#10 +
      'Using Inno Script Studio is especially recommended for new users.';
    Label2Caption := 'Select whether you would like to download and install Inno Script Studio, then click Next.';
    CheckCaption := GetIDECheckCaption('&Download and install Inno Script Studio');

    IDEPage := CreateCustomOptionPage(wpSelectProgramGroup, Caption, SubCaption1, IconFileName, Label1Caption, Label2Caption, CheckCaption, ISStudioCheckBox);

    CreateUrlLabel(IDEPage, ISStudioCheckBox, 0, 'https://www.kymoto.org/inno-script-studio');    

    InnoIDECheckBox := nil;
  end;

  Caption := 'Encryption Support';
  SubCaption1 := 'Would you like to download encryption support?';
  IconFileName := 'ISCrypt.ico';
  Label1Caption :=
    'Inno Setup supports encryption. However, because of encryption import/export laws in some countries, encryption support is not included in the main' +
    ' Inno Setup installer. Instead, it can be downloaded from a server located in the Netherlands now.';
  Label2Caption := 'Select whether you would like to download and install encryption support, then click Next.';
  CheckCaption := '&Download and install encryption support';

  ISCryptPage := CreateCustomOptionPage(IDEPage.ID, Caption, SubCaption1, IconFileName, Label1Caption, Label2Caption, CheckCaption, ISCryptCheckBox);

  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
end;

procedure InitializeWizard;
var
  ISStudioDefault: String;
begin
  CreateCustomPages;

  if PortableCheck then
    ISStudioDefault := '0'
  else
    ISStudioDefault := '1';

  SetInnoIDECheckBoxChecked(GetPreviousData('IDE' {don't change}, '0') = '1');
  ISStudioCheckBox.Checked := GetPreviousData('ISStudio', ISStudioDefault) = '1';
  ISCryptCheckBox.Checked := GetPreviousData('ISCrypt', '1') = '1';

  IDEOrg := GetInnoIDECheckBoxChecked or ISStudioCheckBox.Checked;
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  SetPreviousData(PreviousDataKey, 'IDE' {don't change}, IntToStr(Ord(GetInnoIDECheckBoxChecked)));
  SetPreviousData(PreviousDataKey, 'ISStudio', IntToStr(Ord(ISStudioCheckBox.Checked)));
  SetPreviousData(PreviousDataKey, 'ISCrypt', IntToStr(Ord(ISCryptCheckBox.Checked)));
end;


procedure DownloadFiles(InnoIDE, ISStudio, ISCrypt: Boolean);
begin
  DownloadPage.Clear;
  if InnoIDE then
    DownloadPage.Add('https://jrsoftware.org/download.php/innoide.exe', 'innoide-setup.exe', '');
  if ISStudio then
    DownloadPage.Add('https://jrsoftware.org/download.php/isstudio.exe', 'isstudio-setup.exe', '');
  if ISCrypt then
    DownloadPage.Add('https://jrsoftware.org/download.php/iscrypt.dll', 'ISCrypt.dll', '2f6294f9aa09f59a574b5dcd33be54e16b39377984f3d5658cda44950fa0f8fc');
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      FilesDownloaded := True;
    except
      Log(GetExceptionMessage);
      FilesDownloaded := False;
    end;
  finally
    DownloadPage.Hide;
  end;

  if not FilesDownloaded then
    SuppressibleMsgBox('Setup could not download the extra files. Try again later or download and install the extra files manually.' + #13#13 + 'Setup will now continue installing normally.', mbError, mb_Ok, idOk);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  if CurPageID = wpReady then
    if GetInnoIDECheckBoxChecked or ISStudioCheckBox.Checked or ISCryptCheckBox.Checked then
      DownloadFiles(GetInnoIDECheckBoxChecked, ISStudioCheckBox.Checked, ISCryptCheckBox.Checked);
  Result := True;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := Modifying and ((PageID = wpSelectDir) or (PageID = wpSelectProgramGroup) or ((PageID = IDEPage.ID) and IDEOrg));
end;

function ModifyingCheck: Boolean;
begin
  Result := Modifying;
end;

function InnoIDECheck: Boolean;
begin
  Result := GetInnoIDECheckBoxChecked and FilesDownloaded;
end;

function ISStudioCheck: Boolean;
begin
  Result := ISStudioCheckBox.Checked and FilesDownloaded;
end;

function AnyIDECheck: Boolean;
begin
  Result := InnoIDECheck or ISStudioCheck;
end;

function ISCryptCheck: Boolean;
begin
  Result := ISCryptCheckBox.Checked and FilesDownloaded;
end;

function GetIDEPath(Key, Name: String; var IDEPath: String; var IDEPathRead: Boolean): String;
var
  IDEPathKeyName, IDEPathValueName: String;
begin
  if not IDEPathRead then begin
    IDEPathKeyName := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\' + Key;
    IDEPathValueName := 'Inno Setup: App Path';

    if not RegQueryStringValue(HKLM, IDEPathKeyName, IDEPathValueName, IDEPath) then begin
      if not RegQueryStringValue(HKCU, IDEPathKeyName, IDEPathValueName, IDEPath) then begin
        SuppressibleMsgBox('Error launching InnoIDE:'#13'Could not read InnoIDE path from registry.', mbError, mb_Ok, idOk);
        IDEPath := '';
      end;
    end;

    IDEPathRead := True;
  end;

  Result := IDEPath;
end;

function GetInnoIDEPath(S: String): String;
begin
  Result := GetIDEPath('{1E8BAA74-62A9-421D-A61F-164C7C3943E9}_is1', 'InnoIDE', InnoIDEPath, InnoIDEPathRead);
end;

function GetISStudioPath(S: String): String;
begin
  Result := GetIDEPath('{7C22BD69-9939-43CE-B16E-437DB2A39492}_is1', 'Inno Script Studio', ISStudioPath, ISStudioPathRead);
end;

function GetISStudioCmdLine(S: String): String;
begin
  Result := '/verysilent /group="' + ExpandConstant('{groupname}') + '\Inno Script Studio" /mergetasks="';
  if not WizardIsTaskSelected('desktopicon') then
    Result := Result + '!';
  Result := Result + 'desktopicon,issfileassociation"';
  if PortableCheck then
    Result := Result + ' /portable=1';
end;