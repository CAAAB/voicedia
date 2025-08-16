import os
import torch
import whisper
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from scipy.spatial.distance import cdist
import numpy as np
import warnings

# Suppress user warnings from Whisper
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
ENROLLMENT_DIR = "enrollment_samples"
AUDIO_TO_DIARIZE = "meeting_audio.wav"
WHISPER_MODEL = "base" # "tiny", "base", "small", "medium", "large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---
def format_time(seconds):
    """Converts seconds to a HH:MM:SS.ms string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# --- Main Logic ---

# 1. Load Models
print("Loading models...")
# Whisper for transcription
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
# SpeechBrain for voice fingerprinting (embeddings)
embedding_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)
print("Models loaded.")

# 2. Enrollment: Create voice fingerprints for known speakers
print("\nProcessing enrollment samples...")
speaker_fingerprints = {}
for file_name in os.listdir(ENROLLMENT_DIR):
    speaker_name = os.path.splitext(file_name)[0]
    file_path = os.path.join(ENROLLMENT_DIR, file_name)
    
    try:
        # Load audio, resample to 16kHz as required by SpeechBrain
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Generate the embedding (voice fingerprint)
        with torch.no_grad():
            embedding = embedding_model.encode_batch(waveform)
            embedding = embedding.squeeze(0).squeeze(0) # Make it 1D
        
        speaker_fingerprints[speaker_name] = embedding
        print(f"  - Created fingerprint for: {speaker_name}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if not speaker_fingerprints:
    print("No enrollment samples found or processed. Exiting.")
    exit()

# 3. Transcription: Get text and timestamps from Whisper
print(f"\nTranscribing {AUDIO_TO_DIARIZE} with Whisper...")
# Using word_timestamps=True gives more granular segments
transcription_result = whisper_model.transcribe(AUDIO_TO_DIARIZE, word_timestamps=True)
segments = transcription_result['segments']
print("Transcription complete.")

# 4. Diarization: Match segments to speakers
print("\nDiarizing transcribed segments...")
# Load the full audio file for segment extraction
full_audio, sr = torchaudio.load(AUDIO_TO_DIARIZE)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    full_audio = resampler(full_audio)

diarized_segments = []

# Prepare speaker fingerprint matrix for efficient comparison
known_speakers = list(speaker_fingerprints.keys())
known_embeddings = torch.stack(list(speaker_fingerprints.values())).to(DEVICE)

for segment in segments:
    start_time = segment['start']
    end_time = segment['end']
    text = segment['text']

    # Extract audio chunk for the current segment
    start_sample = int(start_time * 16000)
    end_sample = int(end_time * 16000)
    segment_audio = full_audio[:, start_sample:end_sample]

    if segment_audio.shape[1] == 0: # Skip empty segments
        continue

    # Generate embedding for the segment
    with torch.no_grad():
        segment_embedding = embedding_model.encode_batch(segment_audio.to(DEVICE))
        segment_embedding = segment_embedding.squeeze(0)

    # Compare the segment's embedding to all known speaker fingerprints
    # We use cosine distance (1 - cosine_similarity)
    distances = cdist(segment_embedding.cpu().numpy(), known_embeddings.cpu().numpy(), metric='cosine')
    closest_speaker_index = np.argmin(distances)
    
    # Assign the speaker with the most similar voice
    assigned_speaker = known_speakers[closest_speaker_index]
    
    diarized_segments.append({
        "speaker": assigned_speaker,
        "start": start_time,
        "end": end_time,
        "text": text
    })

print("Diarization complete.")

# 5. Format and Print the Final Output (Corrected)
print("\n--- Final Diarized Transcript ---")

# Check if there are any segments to process
if not diarized_segments:
    print("No speech detected or diarized.")
else:
    # Initialize with the first segment
    current_speaker = diarized_segments[0]['speaker']
    current_transcript = ""
    current_start_time = diarized_segments[0]['start']

    for segment in diarized_segments:
        speaker = segment['speaker']
        
        # If the speaker is the same, append the text
        if speaker == current_speaker:
            current_transcript += segment['text']
        else:
            # If the speaker changes, print the previous transcript blob
            print(f"[{format_time(current_start_time)}] {current_speaker}:{current_transcript.strip()}")
            
            # And start a new one for the new speaker
            current_speaker = speaker
            current_transcript = segment['text']
            current_start_time = segment['start']

    # Print the very last transcript blob after the loop finishes
    print(f"[{format_time(current_start_time)}] {current_speaker}:{current_transcript.strip()}")

print("\n--- End of Transcript ---")