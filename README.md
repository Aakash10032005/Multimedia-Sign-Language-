# Basic Sign Language Alphabet Demo

A small Windows webcam demo that detects one hand, displays a best-effort
alphabet guess, and speaks stable letters using Windows' built-in speech voice.
It does not require the original project's trained weights, Torch, or translation
services.

## Run

Install Python 3.10 or newer, then open PowerShell in this folder and run:

```powershell
python -m pip install -r requirements.txt
python run.py
```

Allow camera access if Windows asks. On the first launch, the hand detector
downloads its MediaPipe model (about 8 MB). Hold one hand in view. A stable
letter is spoken, and a held letter repeats every three seconds. Press **Q** or
**Esc** to quit.

Speech uses Windows PowerShell and its built-in `System.Speech` voice; no extra
Python speech package is needed.

## Notes

- Letter predictions are best-effort and similar hand shapes can be confused.
- J and Z require movement in ASL. This demo recognizes still hand shapes only.
