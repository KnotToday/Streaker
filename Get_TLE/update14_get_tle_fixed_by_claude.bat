@echo off
setlocal EnableDelayedExpansion EnableExtensions

:: -------------------------------
:: CONFIGURATION
:: -------------------------------
set wifiPrefix=2.4
set baseDir="C:\Users\piein\PythonTrials\Streaker\Get_TLE"
set save_directory=%baseDir%\TLEz
set today=%date:~10,4%%date:~4,2%%date:~7,2%
set runVisible=true

:: Extract and normalize hour
set currentTime=%time:~0,2%
if "%currentTime:~0,1%" == " " set currentTime=%currentTime:~1%

:: Handle 12-hour to 24-hour format
set ampm=%time:~-2%
if /i "%ampm%"=="PM" if %currentTime% LSS 12 set /a currentTime+=12
if /i "%ampm%"=="AM" if %currentTime%==12 set currentTime=00

:: Create per-day log file
set logfile=%baseDir%\Get_TLE_%today%.log

:: -------------------------------
:: START LOGGING
:: -------------------------------
echo %date% %time%: Starting script execution
echo %date% %time%: Starting script execution >> %logfile%

:: Add delay to allow system to wake up fully
echo %date% %time%: Waiting for laptop to fully wake up
echo %date% %time%: Waiting for laptop to fully wake up >> %logfile%
timeout /t 60 /nobreak >> %logfile% 2>&1
echo %date% %time%: Timeout complete. Proceeding >> %logfile%

:: -------------------------------
:: TIME-BASED DOWNLOAD CHECK
:: -------------------------------
if %currentTime% LSS 20 (
    echo %date% %time%: Current time is before 8 PM, proceeding with download >> %logfile%
    goto check_connection
) else (
    echo %date% %time%: Current time is after 8 PM, checking for existing files >> %logfile%

    dir /b "%save_directory%\*.tle" | findstr /i "%today%" >nul
    if errorlevel 1 (
        echo %date% %time%: No TLE files found for today, proceeding with download >> %logfile%
        goto check_connection
    )

    for %%f in ("%save_directory%\%today%*.tle") do (
        call :check_timestamp_after_8pm "%%~tf"
        if errorlevel 1 goto end
    )

    echo %date% %time%: No file found after 8 PM, proceeding with download >> %logfile%
)

goto check_connection

:: -------------------------------
:: CHECK FILE TIMESTAMP (subroutine)
:: -------------------------------
:check_timestamp_after_8pm
rem Extract HH and MM from timestamp string like "05/22/2025 20:15"
set input=%~1
for /f "tokens=2" %%a in ("%input%") do set timepart=%%a
set hour=%timepart:~0,2%
set minute=%timepart:~3,2%
if "%hour:~0,1%"==" " set hour=0%hour:~1,1%
call set fileTime=0%hour%%minute%
call set fileTime=%fileTime:~-4%
if %fileTime% GTR 2000 (
    echo %date% %time%: File for today already exists after 8 PM, skipping download >> %logfile%
    exit /b 1
)
exit /b 0

:: -------------------------------
:: CHECK INTERNET CONNECTION
:: -------------------------------
:check_connection
echo %date% %time%: Checking internet connection >> %logfile%
ping -n 1 8.8.8.8 >nul 2>&1
if errorlevel 1 (
    echo %date% %time%: Not connected to internet, attempting Wi-Fi connection >> %logfile%
    goto find_wifi
) else (
    echo %date% %time%: Internet connection is active, skipping Wi-Fi connection >> %logfile%
    goto run_task
)

:: -------------------------------
:: FIND MATCHING WI-FI PROFILE
:: -------------------------------
:find_wifi
set wifiName=
echo %date% %time%: Searching for Wi-Fi profile starting with "%wifiPrefix%" >> %logfile%
for /f "tokens=*" %%n in ('netsh wlan show profiles ^| findstr /i "^%wifiPrefix%"') do (
    for /f "tokens=2 delims=:" %%a in ("%%n") do (
        set wifiName=%%a
        set wifiName=!wifiName:~1!
        goto connect_wifi
    )
)

echo %date% %time%: No Wi-Fi profile found starting with "%wifiPrefix%" >> %logfile%
goto run_task

:: -------------------------------
:: CONNECT TO WI-FI
:: -------------------------------
:connect_wifi
echo %date% %time%: Connecting to Wi-Fi (!wifiName!) >> %logfile%
netsh wlan connect name="!wifiName!" >> %logfile% 2>&1
if errorlevel 1 (
    echo %date% %time%: Failed to connect to Wi-Fi (!wifiName!) >> %logfile%
    goto end
)
echo %date% %time%: Connected to Wi-Fi (!wifiName!) >> %logfile%
echo %date% %time%: Waiting for connection to establish >> %logfile%
timeout /t 10 /nobreak >> %logfile% 2>&1

:: -------------------------------
:: RUN PYTHON SCRIPT
:: -------------------------------
:run_task
echo %date% %time%: Preparing to run Python script >> %logfile%

if not exist "%baseDir%\get_tle.py" (
    echo %date% %time%: ERROR – Python script not found at %baseDir%\get_tle.py >> %logfile%
    goto end
)

if "%runVisible%"=="true" (
    echo %date% %time%: Launching script in VISIBLE debug mode >> %logfile%
    powershell -WindowStyle Normal -Command "Start-Process cmd -ArgumentList '/k python ""%baseDir%\get_tle.py""' -WindowStyle Normal"
) else (
    echo %date% %time%: Running script in silent mode >> %logfile%
    python "%baseDir%\get_tle.py" >> %logfile% 2>&1
    if errorlevel 1 (
        echo %date% %time%: Main task failed >> %logfile%
        echo ---------------------------------------------------------- >> %logfile%
        type %logfile% | findstr /i "error failed traceback exception" >> %logfile%
        echo ---------------------------------------------------------- >> %logfile%
        goto end
    )
    echo %date% %time%: Main task completed successfully >> %logfile%
)

:: -------------------------------
:: END
:: -------------------------------
:end
echo %date% %time%: Ending script execution >> %logfile%
endlocal
exit /b 0
