call .\postprocessing-build.bat

..\..\MVVENV\Scripts\pyinstaller --onefile .\MusicVisualiserV3.py
copy /Y ".\dist\MusicVisualiserV3.exe" ".\MusicVisualiserV3.exe"
copy /Y ".\dist\MusicVisualiserV3.exe" "..\..\..\..\MusicVisualiserOutput\MusicVisualiserV3.exe"