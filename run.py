"""Run the simple webcam based ASL alphabet recognizer with ``python run.py``."""

import sys
import subprocess
import time

import cv2

from src.detection.mediapipe_detector import HandDetector
from src.models.fallback_recognizer import FallbackRecognizer


class LetterAnnouncer:
    """Use Windows' built-in System.Speech voice for each announcement."""

    def announce(self, letter: str) -> None:
        # The classifier only emits single uppercase labels. Keep the input
        # restricted before placing it in a PowerShell command.
        if len(letter) != 1 or not letter.isascii() or not letter.isalpha():
            return
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.Speak('Letter {letter.upper()}'); $voice.Dispose()"
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                creationflags=flags,
            )
        except OSError as exc:
            print(f"Could not start Windows speech: {exc}")

    def close(self) -> None:
        pass


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Close other camera apps and try again.")
        return 1

    detector = None
    announcer = LetterAnnouncer()
    recognizer = FallbackRecognizer()
    candidate_label = ""
    candidate_frames = 0
    last_spoken_label = ""
    last_spoken_at = 0.0
    print("Show one hand to the camera. Stable letters are spoken aloud.")
    print("Press Q or Esc to quit.")
    try:
        detector = HandDetector(max_num_hands=1)
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read from webcam.")
                return 1

            # MediaPipe expects RGB; OpenCV captures BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection = detector.detect(rgb_frame)
            label = ""
            if detection.landmarks_per_hand:
                label, _ = recognizer.predict(detection.landmarks_per_hand[0])

            # A few matching frames reject single-frame noise without missing
            # quick sign changes.
            if label and label == candidate_label:
                candidate_frames += 1
            else:
                candidate_label = label
                candidate_frames = 1 if label else 0
            now = time.monotonic()
            if (candidate_frames >= 3 and candidate_label and
                    (candidate_label != last_spoken_label or now - last_spoken_at >= 3.0)):
                print(f"Speaking: {candidate_label}")
                announcer.announce(candidate_label)
                last_spoken_label = candidate_label
                last_spoken_at = now

            display = cv2.cvtColor(detection.annotated_frame, cv2.COLOR_RGB2BGR)
            text = f"Sign: {label}" if label else "Show a letter sign"
            cv2.rectangle(display, (10, 10), (310, 62), (20, 20, 20), -1)
            cv2.putText(display, text, (22, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (80, 255, 150), 2, cv2.LINE_AA)
            cv2.putText(display, "Q / Esc: quit", (22, display.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        1, cv2.LINE_AA)
            cv2.imshow("ASL Alphabet Recognizer", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Could not start the hand detector: {exc}")
        print("On first run, check your internet connection so MediaPipe can download its model.")
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
        announcer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
