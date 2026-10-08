; GPU压图 安装程序脚本(Inno Setup 7)
; 构建: C:\InnoSetup\ISCC.exe installer.iss

#define AppName "GPU压图"
#define AppVersion "2.1"
#define AppPublisher "GPU压图"
#define AppExe "GPU压图.exe"

[Setup]
AppId={{8E7B6C2A-4D5E-4F6A-9B3C-1A2D3E4F5A6B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
OutputDir=installer
OutputBaseFilename=GPU压图-安装程序

Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
; 语言
ShowLanguageDialog=no

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[CustomMessages]
chinesesimplified.CreateDesktopIcon=创建桌面快捷方式(&D)
chinesesimplified.LaunchProgram=运行 {#AppName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "附加任务:"
Name: "assocpng"; Description: "添加到 PNG 右键菜单(""用GPU压图压缩"")"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "dist\GPU压图\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Classes\System.FileAssociations\image\shell\GPU压图"; ValueType: string; ValueName: ""; ValueData: "用GPU压图压缩"; Tasks: assocpng; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\System.FileAssociations\image\shell\GPU压图\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" --src ""%1"""; Tasks: assocpng; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  gpuOK: Boolean;
begin
  Result := True;
end;
