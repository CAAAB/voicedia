# Voice Diarization

This project performs voice diarization on an audio file. It uses OpenAI's Whisper model for transcription and SpeechBrain for speaker recognition.

## Usage

1.  Install the dependencies:
    ```pip install -r requirements.txt```
2.  Place your enrollment samples (wav files of known speakers) in the `enrollment_samples` directory.
3.  Place the audio file you want to diarize in the root directory and name it `meeting_audio.wav`.
4.  Run the script:
    ```python diarize.py```
