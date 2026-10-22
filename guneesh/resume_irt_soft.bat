@echo off
rem Restarts the version B "soft" IRT fits after a shutdown. Finished fits (DONE) are skipped;
rem an unfinished fit continues from its last saved checkpoint (saved every 200 steps).
cd /d "%~dp0"
python pipeline\irt_fit.py --run versionB_4parts_soft --cont soft --fold all >> logs\irt_B_soft.log 2>&1
