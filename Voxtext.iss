; ============================================================================
; Voxtext - Inno Setup Script
; ============================================================================
; Builds VoxtextSetup-1.0.exe from the PyInstaller onedir output.
;
; Assumes this .iss file lives in the project root and TPC's build output
; is at "TPC Builds\Voxtext\dist\Voxtext\" (Voxtext.exe + _internal\ folder).
; If TPC puts the build elsewhere, edit BuildOutput below.
;
; To compile: open in Inno Setup Compiler and hit Build, or from CLI:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" Voxtext.iss
; ============================================================================

#define AppName        "Voxtext"
#define AppVersion     "1.5"
#define AppPublisher   "Don Burnside"
#define AppURL         "https://donburnside.com"
#define AppExeName     "Voxtext.exe"
#define BuildOutput    "TPC Builds\Voxtext"
#define IconFile       "vticon.ico"

[Setup]
; AppId uniquely identifies this app for upgrades/uninstalls.
; KEEP THIS STABLE across versions or Windows will treat updates as new installs.
; (Generate a new one for Voxsmith via Tools → Generate GUID in the Inno IDE.)
AppId={{B7D9F4A2-3E8C-4F1A-9D5B-2A6E8C1F4B7D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}

; Per-user install by default — no UAC prompt, friendlier for indie distribution.
; User can elevate to system-wide via the dialog if they prefer.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes

; 64-bit only — Voxtext requires modern Windows
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Installer output
OutputDir=Installers
OutputBaseFilename=VoxtextSetup-{#AppVersion}
SetupIconFile={#IconFile}
Compression=lzma2/ultra
SolidCompression=yes

; Wizard appearance
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

; Don't show "Welcome" page — speeds up install for users
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Source: pulls every file from the PyInstaller onedir output recursively.
; Source/Voxtext.exe + Source/_internal/** all end up in {app}.
Source: "{#BuildOutput}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch Voxtext after install completes
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
