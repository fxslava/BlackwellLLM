; =============================================================================
;  Blackwell Deployment Suite  --  installer.iss
; -----------------------------------------------------------------------------
;  Inno Setup 6.x script that packs build_staging/ into the shipping installer
;  for "Blackwell Overlay (Live Translation Prosthetic)".
;
;  The [Code] section is a strict, multi-layer HARDWARE AUDIT gate. Before a
;  single byte is written we validate the host against the engine's minimum
;  contract:
;      * 64-bit Windows 10/11              (declarative -- [Setup] directives)
;      * an NVIDIA CUDA-capable GPU        (WMI: Win32_VideoController)
;      * >= 6 GB VRAM for 7B/8B FP8 models (WMI AdapterRAM + registry fallback)
;      * SSD-backed AI-data volume         (WMI: MSFT_PhysicalDisk.MediaType)
;
;  Every WMI path is wrapped in try/except and degrades SAFELY: on a stripped
;  Windows SKU where a provider/namespace is missing we WARN and proceed rather
;  than abort, so we never block a legitimate install on a telemetry gap.
;
;  Build:  ISCC.exe deploy\installer.iss
;  (run deploy\make_staging.py first to populate build_staging\)
; =============================================================================

#define AppName        "Blackwell Overlay"
#define AppExeName     "blackwell_overlay.exe"
#define AppVersion     "1.0.0"
#define AppPublisher   "Blackwell"
#define StagingDir     "..\build_staging"
; WebView2 Evergreen Runtime GUID (client key under EdgeUpdate\Clients).
#define WebView2Guid   "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
; WebView2 bootstrapper filename -- defined here (before [Files]/[Run] use it);
; mirrors WEBVIEW2_BOOTSTRAP in make_staging.py.
#define WebView2Setup  "MicrosoftEdgeWebview2Setup.exe"

[Setup]
AppId={{9C4B7E2A-1D3F-4A6B-8E9C-0B1A0C000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
OutputBaseFilename=BlackwellOverlaySetup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; --- OS + architecture gate (declarative, enforced before InitializeSetup) ---
; 64-bit Windows 10 (10.0) or newer only. x64compatible admits native x64 and
; ARM64 x64-emulation hosts; the CUDA path still requires an NVIDIA dGPU, caught
; below in the WMI audit.
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; WMI Storage namespace + Program Files write both require elevation.
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; ---- application binaries (root of build_staging) ----------------------------
Source: "{#StagingDir}\{#AppExeName}";            DestDir: "{app}"; Flags: ignoreversion
Source: "{#StagingDir}\blackwell_core.dll";       DestDir: "{app}"; Flags: ignoreversion
; ---- DirectStorage runtime (hard load-time dep when built with USE_DIRECT_STORAGE) ---
Source: "{#StagingDir}\dstorage.dll";             DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\dstoragecore.dll";         DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; ---- portable CUDA runtime (exact cudart64_*.dll staged by make_staging.py) ---
Source: "{#StagingDir}\cudart64_*.dll";           DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; ---- optional CUDA math libs (present only if the engine links them) ----------
Source: "{#StagingDir}\cublas64_*.dll";           DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cublasLt64_*.dll";         DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#StagingDir}\cudnn64_*.dll";            DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
; ---- WebView2 frontend assets -> {app}\ui -------------------------------------
Source: "{#StagingDir}\ui\*";                     DestDir: "{app}\ui"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
; ---- WebView2 Evergreen bootstrapper (deployed on-demand in [Run]) -------------
Source: "{#StagingDir}\{#WebView2Setup}";         DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}";                Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";      Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";          Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
; Silently deploy the WebView2 Evergreen Runtime IFF it is not already present.
; NeedsWebView2Runtime() (below) is the [Run] Check; it short-circuits when the
; runtime is registered under HKLM/HKCU EdgeUpdate\Clients.
Filename: "{tmp}\{#WebView2Setup}"; Parameters: "/silent /install"; \
    StatusMsg: "Deploying Microsoft WebView2 Runtime..."; \
    Flags: waituntilterminated runhidden; Check: NeedsWebView2Runtime

; Offer to launch the prosthetic once everything is in place.
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent

; =============================================================================
;  [Code]  --  Pascal Script hardware audit + configuration synthesis
; =============================================================================
[Code]

const
  MIN_VRAM_GB       = 6;          { hard floor advisory for 7B/8B FP8 weights }
  { MSFT_PhysicalDisk.MediaType enumeration (root\Microsoft\Windows\Storage) }
  MEDIA_TYPE_HDD    = 3;
  MEDIA_TYPE_SSD    = 4;
  MEDIA_TYPE_SCM    = 5;          { storage-class memory / NVDIMM -- treat as fast }

var
  AIDataPage: TInputDirWizardPage;   { custom "where do the LLM weights live" page }

{ ------------------------------------------------------------------------- }
{  Low-level helpers                                                         }
{ ------------------------------------------------------------------------- }

{ Connect to a WMI namespace, returning the SWbemServices automation object.
  Raises on failure -- callers MUST wrap in try/except so a missing provider on
  a stripped SKU degrades to "unknown / proceed" instead of crashing setup. }
function WmiConnect(const Namespace: string): Variant;
var
  Locator: Variant;
begin
  Locator := CreateOleObject('WbemScripting.SWbemLocator');
  Result := Locator.ConnectServer('localhost', Namespace);
  { Force impersonation so protected namespaces (Storage) return data. }
  Result.Security_.ImpersonationLevel := 3;   { wbemImpersonationLevelImpersonate }
end;

{ Double every backslash in a Windows path so it is a valid C++/JSON string
  literal body (e.g. C:\Users\AI -> C:\\Users\\AI). Also escapes any stray
  double quote, defensively. }
function JsonEscapePath(const Path: string): string;
begin
  Result := Path;
  StringChangeEx(Result, '\', '\\', True);
  StringChangeEx(Result, '"', '\"', True);
end;

{ ------------------------------------------------------------------------- }
{  Layer 2/3: GPU vendor + VRAM audit (root\CIMV2, Win32_VideoController)     }
{ ------------------------------------------------------------------------- }

{ Authoritative VRAM read. AdapterRAM is a UInt32 and SATURATES at ~4 GB on
  modern cards (a well-known WMI defect), so when it looks capped we fall back to
  the accurate 64-bit registry value HardwareInformation.qwMemorySize under the
  display-adapter class key. Returns best-known VRAM in bytes, or 0 if unknown. }
function ReadVRAMBytesFromRegistry: Int64;
var
  ClassKey, SubKey: string;
  Names: TArrayOfString;
  I: Integer;
  Raw: AnsiString;
  Val, Best: Int64;
  B: Integer;
begin
  Best := 0;
  ClassKey := 'SYSTEM\CurrentControlSet\Control\Class\' +
              '{4d36e968-e325-11ce-bfc1-08002be10318}';   { Display adapters }
  if RegGetSubkeyNames(HKEY_LOCAL_MACHINE, ClassKey, Names) then
  begin
    for I := 0 to GetArrayLength(Names) - 1 do
    begin
      SubKey := ClassKey + '\' + Names[I];
      { qwMemorySize is a REG_QWORD -> read as 8 raw little-endian bytes. }
      if RegQueryBinaryValue(HKEY_LOCAL_MACHINE, SubKey,
                             'HardwareInformation.qwMemorySize', Raw) then
      begin
        if Length(Raw) >= 8 then
        begin
          Val := 0;
          for B := 7 downto 0 do
            Val := (Val shl 8) or (Ord(Raw[B + 1]) and $FF);
          if Val > Best then
            Best := Val;
        end;
      end;
    end;
  end;
  Result := Best;
end;

{ Scans all video controllers. Sets HasNvidia and returns the largest detected
  VRAM in whole GB (0 if indeterminate). Safe on WMI-less builds (returns 0/False
  via the caller's except handler). }
function AuditGpu(var HasNvidia: Boolean): Integer;
var
  Services, ObjSet, Ctrl: Variant;
  I, Count: Integer;
  Name: string;
  AdapterBytes, RegBytes, BestBytes: Int64;
begin
  HasNvidia := False;
  Result := 0;
  BestBytes := 0;

  Services := WmiConnect('root\CIMV2');
  ObjSet := Services.ExecQuery(
    'SELECT Name, AdapterRAM FROM Win32_VideoController');
  Count := ObjSet.Count;

  for I := 0 to Count - 1 do
  begin
    Ctrl := ObjSet.ItemIndex(I);   { SWbemObjectSet.ItemIndex -> SWbemObject }
    if not VarIsNull(Ctrl.Name) then
    begin
      Name := Uppercase(Ctrl.Name);
      if Pos('NVIDIA', Name) > 0 then
        HasNvidia := True;
    end;
    { AdapterRAM is UInt32 (saturates ~4 GB); keep the max across adapters. }
    if not VarIsNull(Ctrl.AdapterRAM) then
    begin
      AdapterBytes := Ctrl.AdapterRAM;
      if AdapterBytes < 0 then
        AdapterBytes := AdapterBytes + Int64(4294967296);   { unwrap signed }
      if AdapterBytes > BestBytes then
        BestBytes := AdapterBytes;
    end;
  end;

  { If AdapterRAM is at/near its 4 GB ceiling it is untrustworthy -- prefer the
    exact 64-bit registry figure whenever it reports more. }
  if BestBytes >= Int64(4293918720) then   { ~3.999 GB -> treated as capped }
  begin
    RegBytes := ReadVRAMBytesFromRegistry;
    if RegBytes > BestBytes then
      BestBytes := RegBytes;
  end;

  Result := Integer(BestBytes div Int64(1073741824));   { bytes -> whole GB }
end;

{ ------------------------------------------------------------------------- }
{  Layer 4: SSD vs HDD latency audit (root\Microsoft\Windows\Storage)        }
{ ------------------------------------------------------------------------- }

{ Resolves the drive letter of a path to its backing physical disk MediaType.
  Chain: path -> drive letter -> MSFT_Partition(DriveLetter) -> DiskNumber ->
  MSFT_PhysicalDisk(DeviceId) -> MediaType. Returns one of MEDIA_TYPE_*, or -1
  when the mapping cannot be resolved (spanned volumes, storage spaces, or a
  Windows SKU without the Storage Management provider). }
function GetDriveMediaType(const Path: string): Integer;
var
  Services, PartSet, Part, DiskSet, Disk: Variant;
  DriveLetter: string;
  DiskNumber, I, J: Integer;
  Found: Boolean;
begin
  Result := -1;
  DriveLetter := Uppercase(Copy(ExtractFileDrive(Path), 1, 1));
  if DriveLetter = '' then
    Exit;

  Services := WmiConnect('root\Microsoft\Windows\Storage');

  { Partition(s) carrying this drive letter -> their owning disk number. }
  { NB: keep the Format() argument array on the same line as the string -- a line
    that STARTS with '[' is parsed by ISCC as a section header ("Invalid section
    tag"), even inside [Code]. }
  PartSet := Services.ExecQuery(Format(
    'SELECT DiskNumber FROM MSFT_Partition WHERE DriveLetter=''%s''', [DriveLetter]));
  if PartSet.Count = 0 then
    Exit;   { unmapped (e.g. network / ReFS storage space) -> unknown }

  Found := False;
  DiskNumber := -1;
  for I := 0 to PartSet.Count - 1 do
  begin
    Part := PartSet.ItemIndex(I);
    if not VarIsNull(Part.DiskNumber) then
    begin
      DiskNumber := Part.DiskNumber;
      Found := True;
      Break;
    end;
  end;
  if not Found then
    Exit;

  { PhysicalDisk whose DeviceId matches the disk number -> its MediaType. }
  DiskSet := Services.ExecQuery(Format(
    'SELECT MediaType FROM MSFT_PhysicalDisk WHERE DeviceId=''%d''', [DiskNumber]));
  for J := 0 to DiskSet.Count - 1 do
  begin
    Disk := DiskSet.ItemIndex(J);
    if not VarIsNull(Disk.MediaType) then
    begin
      Result := Integer(Disk.MediaType);
      Exit;
    end;
  end;
end;

{ Returns True if the user may proceed with the chosen AI-data location.
  Only an explicitly-identified HDD triggers a warning; unknown/SSD/SCM pass
  silently so we never nag on hardware we cannot classify. }
function ConfirmAIStorageMedia(const Path: string): Boolean;
var
  MediaType: Integer;
begin
  Result := True;
  try
    MediaType := GetDriveMediaType(Path);
  except
    { Storage namespace absent on this SKU -> cannot classify -> allow. }
    Log('Storage WMI audit failed: ' + GetExceptionMessage);
    MediaType := -1;
  end;

  if MediaType = MEDIA_TYPE_HDD then
  begin
    Result := (MsgBox(
      'The selected AI-data drive appears to be a mechanical hard disk (HDD).' + #13#10 + #13#10 +
      'Blackwell Overlay streams model weights and performs speculative KV-cache ' +
      'updates from this location on every generation step. On an HDD the seek ' +
      'penalty causes multi-second stalls and visible micro-stutter in the live ' +
      'translation overlay.' + #13#10 + #13#10 +
      'An SSD or NVMe drive is strongly recommended.' + #13#10 + #13#10 +
      'Choose a different location? (Select "No" to force-proceed on this HDD.)',
      mbConfirmation, MB_YESNO) = IDNO);
  end;
end;

{ ------------------------------------------------------------------------- }
{  Wizard lifecycle                                                          }
{ ------------------------------------------------------------------------- }

{ Layer 1 hardware gate. Runs before the wizard is shown; returning False aborts
  setup cleanly. NVIDIA absence is fatal; low VRAM is an informed-consent warning. }
function InitializeSetup: Boolean;
var
  HasNvidia: Boolean;
  VramGb: Integer;
begin
  Result := True;
  HasNvidia := False;
  VramGb := 0;

  { GPU + VRAM audit -- any WMI failure degrades to "cannot verify, proceed". }
  try
    VramGb := AuditGpu(HasNvidia);
  except
    Log('GPU WMI audit failed (' + GetExceptionMessage +
        '); skipping hardware gate.');
    Exit;   { Result stays True: never block install on a broken WMI stack. }
  end;

  { --- NVIDIA presence: hard requirement --- }
  if not HasNvidia then
  begin
    MsgBox(
      'Critical Error: No NVIDIA GPU detected!' + #13#10 + #13#10 +
      'Blackwell Overlay requires a CUDA-compatible graphics card for local ' +
      'inference. Installation will abort.',
      mbCriticalError, MB_OK);
    Result := False;
    Exit;
  end;

  { --- VRAM floor: informed-consent warning --- }
  if (VramGb > 0) and (VramGb < MIN_VRAM_GB) then
  begin
    Result := (MsgBox(
      Format('Low video memory detected: approximately %d GB VRAM.', [VramGb]) + #13#10 + #13#10 +
      'The 7B/8B FP8 models Blackwell Overlay ships with require roughly 6-8 GB ' +
      'of VRAM to load and run without offloading. On this GPU the engine will ' +
      'page weights to system RAM, sharply reducing throughput -- or may fail to ' +
      'load the larger models entirely.' + #13#10 + #13#10 +
      'Continue with the installation anyway?',
      mbConfirmation, MB_YESNO) = IDYES);
  end;
end;

{ Inject the custom AI-data directory page immediately after the standard
  "Select Destination Location" page (wpSelectDir). }
procedure InitializeWizard;
begin
  AIDataPage := CreateInputDirPage(
    wpSelectDir,
    'Select AI Data Location',
    'Where should Blackwell Overlay store model weights and KV-caches?',
    'The engine keeps multi-gigabyte LLM weights and persistent KV-cache/prefix ' +
    'files separate from the application. Choose a fast, roomy volume (SSD/NVMe ' +
    'strongly recommended), then click Next.',
    False, '');
  AIDataPage.Add('');
  AIDataPage.Values[0] := ExpandConstant('{localappdata}\Blackwell\AI_Data');
end;

{ Intercept "Next" on the AI-data page to run the SSD/HDD latency audit. Returning
  False keeps the user on the page so they can pick a faster volume. }
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (AIDataPage <> nil) and (CurPageID = AIDataPage.ID) then
    Result := ConfirmAIStorageMedia(AIDataPage.Values[0]);
end;

{ ------------------------------------------------------------------------- }
{  Post-install: synthesize config.json with the C++-escaped AI data path    }
{ ------------------------------------------------------------------------- }

procedure WriteRuntimeConfig;
var
  ConfigPath, AIData, AIDataEsc: string;
  Json: TArrayOfString;
begin
  AIData := AIDataPage.Values[0];

  { Materialize the AI-data directory now so the engine never races to mkdir it. }
  if not DirExists(AIData) then
    ForceDirectories(AIData);

  AIDataEsc := JsonEscapePath(AIData);
  ConfigPath := ExpandConstant('{app}\config.json');

  { Keys MUST match the schema ConfigStore::Load() parses
    (src/tools/poc_overlay/config.cpp): the app reads "modelPath" (camelCase)
    and "spillFilePath" -- NOT "model_path". A mismatched key is silently
    ignored (nlohmann value()-with-default), leaving the engine with no model
    directory. spillFilePath points at the NVMe KV-spill file inside the chosen
    AI-data volume. }
  SetArrayLength(Json, 4);
  Json[0] := '{';
  Json[1] := '  "modelPath": "'     + AIDataEsc + '",';
  Json[2] := '  "spillFilePath": "' + AIDataEsc + '\\spill.bkv"';
  Json[3] := '}';

  if not SaveStringsToFile(ConfigPath, Json, False) then
    MsgBox('Warning: could not write ' + ConfigPath +
           '. Blackwell Overlay will fall back to its built-in defaults.',
           mbError, MB_OK);
end;

{ ssPostInstall fires once all [Files] have been copied. }
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteRuntimeConfig;
end;

{ ------------------------------------------------------------------------- }
{  [Run] Check: is the WebView2 Evergreen Runtime missing?                   }
{ ------------------------------------------------------------------------- }

{ Probes the three EdgeUpdate\Clients client-key locations (per-machine 64-bit,
  per-machine 32-bit view, and per-user). A non-empty 'pv' version value means
  the runtime is installed. Returns True when it is ABSENT (so [Run] deploys it). }
function WebView2VersionPresent(RootKey: Integer; const SubKey: string): Boolean;
var
  Version: string;
begin
  Result := RegQueryStringValue(RootKey, SubKey, 'pv', Version) and
            (Version <> '') and (Version <> '0.0.0.0');
end;

function NeedsWebView2Runtime: Boolean;
var
  KeyPath: string;
  Installed: Boolean;
begin
  KeyPath := 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{#WebView2Guid}';
  { Installed anywhere -> not needed. HKLM 64-bit, HKLM 32-bit view, HKCU. }
  Installed :=
    WebView2VersionPresent(HKEY_LOCAL_MACHINE, KeyPath) or
    WebView2VersionPresent(HKEY_LOCAL_MACHINE, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{#WebView2Guid}') or
    WebView2VersionPresent(HKEY_CURRENT_USER, KeyPath);

  // Deploy only if the runtime is BOTH absent AND we actually bundled the
  // bootstrapper (make_staging.py stages it opportunistically). Without the file
  // present in {tmp} there is nothing to run, so suppress the [Run] entry rather
  // than fire a "file not found" error.
  // (Line comments here on purpose: a '{...}' constant inside a { } comment would
  //  close the comment early -- Inno's brace comments do not nest.)
  Result := (not Installed) and
            FileExists(ExpandConstant('{tmp}\{#WebView2Setup}'));
end;
