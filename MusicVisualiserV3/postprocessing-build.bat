..\..\MVVENV\Scripts\pyinstaller --onefile .\postprocessing.py
copy /Y ".\dist\postprocessing.exe" "..\..\..\..\MusicVisualiserOutput\postprocessing.exe"
copy /Y ".\dist\postprocessing.exe" ".\MusicVisualiserOutput\postprocessing.exe"