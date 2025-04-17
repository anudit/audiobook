#!/usr/bin/env python3
import torch
try:
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.95) # Keep memory fraction setting
except AttributeError:
    print("MPS backend not available or older PyTorch version. Skipping MPS memory setting.")
except Exception as e:
    print(f"Error setting MPS memory fraction: {e}")


import os
import re
import time
import numpy as np
import soundfile as sf
import fitz # PyMuPDF
from tqdm import tqdm
from kokoro import KPipeline

# Initialize the TTS Pipeline (ensure the model is downloaded)
print("Initializing TTS Pipeline...")
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device="mps")


def read_pdf_pages(file_path):
    """Reads a PDF file and returns a list of text content per page."""
    pages_text = []
    print(f"Reading PDF: {file_path}")
    try:
        with fitz.open(file_path) as doc:
            num_pages = len(doc)
            print(f"Found {num_pages} pages.")
            # Extract text page by page
            for i, page in enumerate(tqdm(doc, desc="Extracting text from pages", unit="page")):
                page_text = page.get_text("text", sort=True) # Get text, maintaining reading order
                pages_text.append(page_text.strip())
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return None
    print(f"Successfully extracted text from {len(pages_text)} pages.")
    return pages_text

# text_to_speech function remains largely the same, processing the text it's given
# Optional: Add leave=False to the inner tqdm if you don't want segment progress bars persisting
def text_to_speech(text, voice='af_heart'):
    """
    Converts the given text (expected to be a chunk) into speech.
    """
    if not text:
        # print("Skipping TTS for empty text chunk.") # Less verbose
        return np.array([])

    # print(f"Starting TTS for chunk ({len(text)} chars)...") # Can be verbose
    audio_segments = []
    try:
        generator = pipeline(text, voice=voice)
        # Inner progress bar for segments within this chunk (optional)
        # Use leave=False so it cleans up after the chunk is done
        # Disable=True can hide it completely if desired
        iterable = tqdm(generator, desc="  Generating segments", unit="seg", leave=False, disable=False)
        for i, (gs, ps, audio_segment) in enumerate(iterable):
            audio_segments.append(audio_segment)
            # else:
            #     tqdm.write(f"Warning: Received empty audio segment at step {i} in chunk")

    except Exception as e:
        print(f"\nError during TTS generation for a chunk: {e}")
        # Optionally log the problematic text chunk here
        # Decide if you want to return partial audio or nothing for this chunk
        # return np.array([]) # Option: return empty for this chunk on error

    # Merge segments for *this chunk*
    if audio_segments:
        try:
            chunk_audio = np.concatenate(audio_segments)
        except ValueError as ve:
             print(f"Error concatenating segments within a chunk: {ve}.")
             valid_segments = [seg for seg in audio_segments if seg is not None and seg.size > 0]
             if valid_segments:
                 chunk_audio = np.concatenate(valid_segments)
             else:
                 chunk_audio = np.array([])
        except Exception as e:
             print(f"Unexpected error during chunk audio concatenation: {e}")
             chunk_audio = np.array([])
    else:
        # print("\nNo audio segments generated for this chunk.") # Can be verbose
        chunk_audio = np.array([])

    return chunk_audio

def main():
    # --- Configuration ---
    input_pdf = 'book.pdf'
    output_dir = "./output_audio"
    tts_voice = 'af_heart' # Using the Afrikaans voice from previous example
    chunk_size = 10 # Process in chunks of 10 pages
    # --- End Configuration ---

    # if not os.path.exists(input_pdf):
    #     print(f"Error: Input PDF file not found at '{input_pdf}'")
    #     return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Read all text from the PDF, page by page
    pages_text = read_pdf_pages(input_pdf)

    if not pages_text: # Check if text extraction was successful
        print("Text extraction failed or PDF contains no text. Exiting.")
        return

    # 2. Create chunks of pages
    # Group the list of page texts into chunks of 'chunk_size'
    chunks = [pages_text[i:i + chunk_size] for i in range(0, len(pages_text), chunk_size)]
    num_chunks = len(chunks)
    total_pages = len(pages_text)
    print(f"Divided {total_pages} pages into {num_chunks} chunks of up to {chunk_size} pages each.")

    all_chunk_audio = []
    start_time = time.time()

    # 3. Process each chunk with a progress bar
    print("Starting chunked text-to-speech conversion...")
    # This tqdm loop tracks progress over chunks
    for i, chunk_pages in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
        # Combine text for the current chunk
        # Use double newline as a separator, similar to the original full text approach
        chunk_text = "\n\n".join(chunk_pages)
        # Apply basic cleaning to the combined chunk text
        chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()

        if not chunk_text or len(chunk_text) < 5: # Skip very small/empty chunks
             # tqdm.write(f"Skipping empty or very short chunk {i+1}/{num_chunks}")
             continue

        # Generate audio for this chunk
        chunk_audio = text_to_speech(chunk_text, voice=tts_voice)

        # Collect the generated audio
        if chunk_audio.size > 0:
            sf.write(f"./output_audio/chunk-{i}.wav", chunk_audio, 24000)
            all_chunk_audio.append(chunk_audio)
        else:
             tqdm.write(f"Warning: Chunk {i+1}/{num_chunks} produced no audio.")


    elapsed = time.time() - start_time
    print(f"\nFinished processing all {num_chunks} chunks in {elapsed:.2f} seconds.")

    # 4. Concatenate audio from all chunks
    if not all_chunk_audio:
        print("No audio was generated from any chunk. Cannot save file.")
        return

    print("Concatenating audio from all chunks...")
    try:
        full_audio = np.concatenate(all_chunk_audio)
        print("Final audio concatenation complete.")
    except ValueError as ve:
        print(f"Error concatenating final audio: {ve}. Attempting to filter invalid chunks...")
        valid_final_segments = [seg for seg in all_chunk_audio if seg is not None and seg.size > 0]
        if valid_final_segments:
             print("Retrying final concatenation with valid chunk audio...")
             full_audio = np.concatenate(valid_final_segments)
        else:
             print("No valid chunk audio found for final concatenation.")
             full_audio = np.array([])
    except Exception as e:
        print(f"Unexpected error during final audio concatenation: {e}")
        full_audio = np.array([])

    if full_audio.size == 0:
        print("Final audio is empty after concatenation. No file saved.")
        return

    # 5. Save the single concatenated audio file
    pdf_basename = os.path.basename(input_pdf)
    safe_title = re.sub(r'[^\w\s-]', '', os.path.splitext(pdf_basename)[0]).strip().replace(' ', '_')
    if not safe_title:
        safe_title = "output_audio"
    output_filename = os.path.join(output_dir, f"{safe_title}_chunked.wav") # Added suffix

    print(f"\nSaving final concatenated audio to {output_filename}...")
    try:
        sf.write(output_filename, full_audio, 24000)
        duration_seconds = len(full_audio) / 24000
        print(f"Successfully saved final audio file.")
        # Total processing time was already printed after chunk loop
        print(f"Generated audio duration: {duration_seconds:.2f} seconds ({duration_seconds / 60:.2f} minutes).")
    except Exception as e:
        print(f"Error saving final audio file {output_filename}: {e}")


if __name__ == "__main__":
    main()
