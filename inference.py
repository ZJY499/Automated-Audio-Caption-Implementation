from __future__ import annotations
import argparse
import warnings

import pandas as pd
from pathlib import Path
import soundfile
import torch
import librosa
from audidata.datasets import Clotho
from audidata.io.crops import RandomCrop
from audidata.transforms import Mono

from data.text_normalization import TextNormalization
from data.text_tokenization import BertTokenizer
from train import get_audio_encoder, get_llm_decoder, get_audio_latent


def inference(audio_path: str, num_samples: int = 1):
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Arguments
    ckpt_path = r"D:\Y3ACCsurf\audio_caption\mini_audio_caption\step=20000.pth"  # args.ckpt_path

    # Default parameters
    sr = 32000  # To be consistent with the encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 30  # Max caption length
    clip_duration = 10.  # Audio clip duration
    audio_encoder_name = "Cnn14"
    llm_decoder_name = "Llama"
    temperature = 1.0
    top_k = 200

    # Audio Cropper
    crop = RandomCrop(clip_duration=clip_duration, end_pad=0.)

    # Caption transforms
    target_transform = [
        TextNormalization(),  # Remove punctuations
        BertTokenizer(max_length=max_length)  # Convert captions to token IDs
    ]
    tokenizer = target_transform[1].tokenizer
    start_token_id = tokenizer.cls_token_id  # 101
    text_vocab_size = tokenizer.vocab_size  # 30,522

    # Load audio encoder
    audio_encoder, audio_latent_dim = get_audio_encoder(model_name=audio_encoder_name)
    audio_encoder.to(device)

    # Load LLM decoder with proper device mapping
    llm_decoder = get_llm_decoder(
        model_name=llm_decoder_name, 
        audio_latent_dim=audio_latent_dim, 
        text_vocab_size=text_vocab_size
    )
    
    # Load checkpoint with proper device mapping
    checkpoint = torch.load(ckpt_path, map_location=device)
    llm_decoder.load_state_dict(checkpoint)
    llm_decoder.to(device)

    # Text start token
    text_ids = torch.LongTensor([[start_token_id]]).to(device)  # (b, 1)

    # Load and process audio
    audio, _ = librosa.load(path=audio_path, sr=sr, mono=True)
    audio = torch.Tensor(audio[None, None, :]).to(device)  # shape: (b, c, t_audio)
    
    # Extract audio embeddings
    audio_latent = get_audio_latent(
        model_name=audio_encoder_name, 
        model=audio_encoder, 
        audio=audio
    )

    predictions = []
    # Sample    
    for n in range(num_samples):
        # Combine audio embeddings and text ids
        input_seqs = [audio_latent, text_ids]
        seq_types = ["audio", "text"]

        with torch.no_grad():
            llm_decoder.eval()
        
            outputs = llm_decoder.generate(
                seqs=input_seqs,
                seq_types=seq_types,
                max_new_tokens=max_length, 
                temperature=temperature, 
                top_k=top_k
            )

        sampled_text_ids = outputs[-1][0].cpu().numpy()
        strings = tokenizer.decode(token_ids=sampled_text_ids, skip_special_tokens=True)
        predictions.append(strings)
        print("Prediction: {}".format(strings))

    return predictions


def get_clotho_meta(root: str, split: str) -> dict:
    r"""Load Clotho audio paths and captions."""
    if split == "train":
        meta_csv = Path(root, "clotho_captions_development.csv")
        audios_dir = Path(root, "clotho_audio_development")

    elif split == "test":
        meta_csv = Path(root, "clotho_captions_evaluation.csv")
        audios_dir = Path(root, "clotho_audio_evaluation")

    else:
        raise ValueError(split)

    meta_dict = {
        "audio_name": [],
        "audio_path": [],
        "captions": []
    }

    df = pd.read_csv(meta_csv, sep=',')

    for n in range(len(df)):
        meta_dict["audio_name"].append(df["file_name"][n])
        meta_dict["audio_path"].append(Path(audios_dir, df["file_name"][n]))
        meta_dict["captions"].append([df["caption_{}".format(i)][n] for i in range(1, 6)])

    return meta_dict


if __name__ == "__main__":
    audio_path = r'D:\Y3ACCsurf\audio_caption\mini_audio_caption\assets\young_artists.wav'
    inference(audio_path, num_samples=1)