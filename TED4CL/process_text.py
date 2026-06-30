from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import os


LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "italian": "it",
    "japanese": "ja",
    "turkish": "tr",
    "hindi": "hi",
}


def _find_subtitle_file(path, video_name, lang_code):
    video_dir = os.path.join(path, video_name)
    exact_subtitle = os.path.join(video_dir, f"{video_name}_subtitles_{lang_code}.txt")
    if os.path.isfile(exact_subtitle):
        return exact_subtitle

    if not os.path.isdir(video_dir):
        return None

    subtitle_prefix = f"{video_name}_subtitles_"
    for filename in sorted(os.listdir(video_dir)):
        if not filename.startswith(subtitle_prefix) or not filename.endswith(".txt"):
            continue
        saved_lang = filename[len(subtitle_prefix):-4].lower().replace("_", "-")
        if saved_lang.split("-")[0] == lang_code:
            return os.path.join(video_dir, filename)

    return None


# If subs are available in the original language, this function process them, otherwise process the english version
def select_sub_from_language(path, video_name, language):
    # Get the language code from the language parameter
    language_key = str(language).lower().replace("_", "-")
    lang_code = LANGUAGE_NAME_TO_CODE.get(language_key)
    if not lang_code and language_key in set(LANGUAGE_NAME_TO_CODE.values()):
        lang_code = language_key

    if not lang_code:
        raise ValueError(f"Unsupported language '{language}'")

    original_subtitle = _find_subtitle_file(path, video_name, lang_code)
    english_subtitle = _find_subtitle_file(path, video_name, "en")

    #print("HERE", original_subtitle)

    # Check if the original language subtitle file exists
    if original_subtitle:
        return original_subtitle, lang_code
    # If not, check for English subtitles as a fallback
    elif english_subtitle:
        return english_subtitle, "en"
    else:
        raise FileNotFoundError(f"No subtitles found for '{language}' or English.")

# Collects the text from start_time to end_time from the subtitles.
# Note that if a word starts before the start_time but ends after the start_time, it will be included.
def extract_sample_text(start_time, end_time, subs):

    collected_text = []

    for line in subs:
        # Skip empty lines
        #print("LINE",line)
        if not line.strip():
            continue

        # Regex pattern to extract times and text
        pattern = r'^([\d.]+)\s*-\s*([\d.]+):\s*(.*)$'
        match = re.match(pattern, line)

        if match:
            text_start = float(match.group(1))
            duration = float(match.group(2))
            text = match.group(3).strip()
            text_end = text_start + duration

            if text_start > end_time:
                break

            # Check for overlap with the desired time range
            if text_start <= end_time and text_end >= start_time:
                collected_text.append(text)


    # Combine all collected texts into one string
    final_sentence = ' '.join(collected_text)

    return final_sentence


# Function to encode the text with LaBSE
def encode_text_with_labse(text,tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output  # Use pooled output for sentence-level embedding
    embeddings = embeddings.detach().cpu().numpy() #convert in numpy
    return embeddings

# Load subtitles
def load_subs(path, video_name, language):
    sub_file, lang_code = select_sub_from_language(path, video_name, language)
    with open(sub_file,"r",encoding="utf-8") as sf:
        subs = sf.readlines()
    return subs, lang_code

#For test
if __name__ == '__main__':
    # Load LaBSE model and tokenizer
    model_name = 'sentence-transformers/LaBSE'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Read subtitles from the uploaded Hindi file
    path = r"PATH/TO/SPECIFIC/PLAYLIST"
    directory, filename = os.path.split(path)
    language = filename.split("_")[0]

    video_name = "_9Hw2vuQ75M"
    start_time = 20
    end_time = 40

    sub_file = load_subs(path, video_name, language)
    subs = extract_sample_text(start_time, end_time, sub_file)
    subs_encoding = encode_text_with_labse(subs,tokenizer,model)

    translated_sentence = "Questo bhajan è per mia madre. Karan, ho avuto la fortuna di essere qui in piedi. Saluti."
    subs_encoding_translated = encode_text_with_labse(translated_sentence, tokenizer, model)

    similarity = cosine_similarity(subs_encoding_translated, subs_encoding)

    print("Subs",subs)
    print("Encoding",subs_encoding.shape,subs_encoding)
    print("Similarity", similarity)
