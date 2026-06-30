import torch
import torch.nn.functional as F
import librosa.filters
import soundfile as sf
from scipy.signal import resample
import numpy as np
import os
import pyloudnorm as pyln
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model_name = "voidful/wav2vec2-xlsr-multilingual-56"
processor = None
model = None


def load_default_wav2vec_model(device=None):
    """Load the default multilingual Wav2Vec2 model on demand."""
    global processor, model
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    if model is None:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        if device is not None:
            model.to(device)
        model.eval()
    return model, processor

# given a duration times in second, find how many wav2vec samples correspond to it
def calc_wav2vec2_output_frames(duration_in_seconds, sampling_rate, downsampling_factor):
    output_frame_rate = sampling_rate / downsampling_factor
    output_frames = duration_in_seconds * output_frame_rate
    return int(round(output_frames)) - 1

# estimates the spectogram length depending on length of motion data (useful to check if audio and motion are syncronized
# and to find the mel values corresponding to motion)
def calc_spectrogram_length_from_motion_length(poses_frames, poses_fps, sample_rate, hop_length):
    hop_duration = hop_length / sample_rate  # Duration per Mel frame in seconds
    time_in_seconds = poses_frames / poses_fps  # Convert motion frame to time
    mel_index = int(time_in_seconds / hop_duration)
    return mel_index

# estimates the spectogram length depending on time in seconds
def calc_spectrogram_length_from_time(time_in_sec, sample_rate, hop_length):
    hop_duration = hop_length / sample_rate
    mel_index = int(time_in_sec / hop_duration)
    return mel_index


def extract_wav2vec_embeddings(audio, model, processor, sample_rate=16000):
    """
    Extract Wav2Vec2 embeddings from raw audio input.

    Parameters:
    - audio (np.ndarray): Raw audio input (1D array) at 16000 Hz sample rate.
    - sample_rate (int): Sample rate of the audio input. Default is 16000 Hz.

    Returns:
    - embeddings (torch.Tensor): Extracted embeddings of shape (num_frames, embedding_dim).
    """
    # Check if the sample rate is 16000 Hz, as required by Wav2Vec2
    if sample_rate != 16000:
        raise ValueError("The audio input must have a sample rate of 16000 Hz.")

    # Process the audio with the Wav2Vec2 processor to prepare it for the model
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Extract embeddings by passing the inputs through the model
    with torch.no_grad():  # Disable gradients for inference
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the embeddings from the hidden states (e.g., last hidden layer)
    embeddings = outputs.hidden_states[-1].squeeze(0)  # (num_frames, embedding_dim)

    return embeddings


def extract_mfcc(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs  # 2D array of MFCC values (n_mfcc, frames)


def extract_mel_log(audio, sample_rate, n_mels=64):
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate,  n_fft=1024, hop_length=512, power=2, n_mels=n_mels)

    # Convert to log scale (dB) for perceptual loudness
    mel_log = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_log  # 2D array of Mel-log values (n_mels, frames)

def extract_spectral_flux(audio, sample_rate):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    spectral_flux = np.diff(onset_env, prepend=onset_env[0])
    return spectral_flux  # Array of spectral flux values


def extract_formants(audio, sample_rate):
    snd = parselmouth.Sound(audio, sampling_frequency=sample_rate)
    formant = snd.to_formant_burg()
    formants = []

    for t in range(len(audio) // sample_rate):
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        formants.append((f1, f2))

    return formants  # List of tuples (F1, F2) values

def extract_pitch(audio, sample_rate):
    pitch = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return pitch  # Array of pitch values for each frame

def extract_onsets(audio, sample_rate):
    onset_strength = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    return onset_strength   # Array of times where onsets occur

# load audio file, resample to 16000 Hz, remove noise, normalize loudness, and save as npy
def load_audio(path, video_name,target_sr = 16000):
    # TODO some files appear as .mp3, some other as .mp3.mp3. Solve this issue during download phase
    audio_file = video_name + "_audio.mp3"
    audio_file2 = video_name + "_audio.mp3.mp3"
    audio_file_denoised = video_name + "_audio_denoised.npy"
    audio_denoised_path = os.path.join(path,video_name,audio_file_denoised)

    if os.path.exists(audio_denoised_path):
        audio_data = np.load(audio_denoised_path)
    else:
        if os.path.exists(os.path.join(path,video_name,audio_file)):
            try:
                audio, sr = librosa.load(os.path.join(path,video_name,audio_file),sr=target_sr)
            except Exception as e:
                print(f"Error with librosa: {e}")
                try:
                    # Fallback to soundfile
                    audio, sr = sf.read(os.path.join(path,video_name,audio_file))
                except Exception as e_sf:
                    print(f"Error with soundfile: {e_sf}")
                    audio, sr = None, None  # Mark as failed
        elif os.path.exists(os.path.join(path,video_name,audio_file2)):
            try:
                audio, sr = librosa.load(os.path.join(path, video_name, audio_file2),sr=target_sr)
                print("Loaded audio file v2...")
            except Exception as e:
                print(f"Error with librosa: {e}")
                try:
                    # Fallback to soundfile
                    audio, sr = sf.read(os.path.join(path, video_name, audio_file2))
                except Exception as e_sf:
                    print(f"Error with soundfile: {e_sf}")
                    audio, sr = None, None  # Mark as failed
        else:
            raise ValueError(f"No file audio {audio_file} found")
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr) #resample to 16000
        meter = pyln.Meter(target_sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, -20.0) #normalize loudness
        audio_denoised = librosa.effects.preemphasis(audio)  # remove noise
        np.save(audio_denoised_path, audio_denoised)
        audio_data = audio_denoised

    return audio_data
