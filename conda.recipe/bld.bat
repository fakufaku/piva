if "%ARCH%" == "32" (set PLATFORM=x86) else (set PLATFORM=x64)
call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" %PLATFORM%
set DISTUTILS_USE_SDK=1
set MSSdk=1
"%PYTHON%" setup.py install
if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

:: See
:: http://docs.continuum.io/conda/build.html
:: for a list of environment variables that are set during the build process.
