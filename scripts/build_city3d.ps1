param(
    [string]$CondaEnvironment = "urbanstock-city3d",
    [string]$City3DSource = ".tools\City3D"
)

$ErrorActionPreference = "Stop"
$ExpectedCommit = "c9299efe61625f03a78245683eaa155a9670df0e"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $City3DSource))
$WrapperPath = Join-Path $ProjectRoot "native\city3d_cli\main.cpp"
$TargetMain = Join-Path $SourcePath "code\CLI_Example_1\main.cpp"
$BuildPath = Join-Path $SourcePath "build"
$Conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"

if (-not (Test-Path -LiteralPath $SourcePath -PathType Container)) {
    throw "City3D source is missing at $SourcePath"
}
$ActualCommit = (& git -C $SourcePath rev-parse HEAD).Trim()
if ($ActualCommit -ne $ExpectedCommit) {
    throw "Expected City3D commit $ExpectedCommit, found $ActualCommit"
}
Copy-Item -LiteralPath $WrapperPath -Destination $TargetMain -Force

& $Conda run -n $CondaEnvironment cmake -S $SourcePath -B $BuildPath -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "City3D CMake configuration failed" }
& $Conda run -n $CondaEnvironment cmake --build $BuildPath --target CLI_Example_1 --config Release --parallel 8
if ($LASTEXITCODE -ne 0) { throw "City3D compilation failed" }

$Executable = Join-Path $BuildPath "bin\CLI_Example_1.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "City3D executable was not generated"
}
Write-Output $Executable
