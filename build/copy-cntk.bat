@rem Copy CNTK from a local build to third_party/cntk
setlocal
set BUILD=Release
if '%1'=='debug' (set BUILD=Debug)
if '%1'=='Debug' (set BUILD=Debug)
set SRC=c:\dev\tools\cntk
set SRCBIN=%SRC%\x64\%BUILD%
set DST=third_party\cntk\Windows
mkdir third_party\cntk
mkdir %DST%
mkdir %DST%\lib
mkdir %DST%\include
xcopy /y %SRCBIN%\*.lib %DST%\lib\
xcopy /y %SRCBIN%\*.dll %DST%\lib\
xcopy /y %SRC%\Source\CNTKv2LibraryDll\API\* %DST%\include\