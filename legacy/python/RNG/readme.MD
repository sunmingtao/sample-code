Package to an exe file

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install pillow
pip install pyinstaller
pyinstaller --onefile --windowed rng.py
```

Copy the rng.exe, white.ico and white.png to the same folder.  