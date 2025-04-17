#!/usr/bin/env python3
import torch
torch.mps.set_per_process_memory_fraction(0.95)

import os
import re
import time
import numpy as np
import soundfile as sf
from tqdm import tqdm
from ebooklib import epub
from bs4 import BeautifulSoup
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device="mps")

def read_epub(file_path):
    book = epub.read_epub(file_path)
    chapters = []

    # Iterate over all items in the EPUB. We only want those which are HTML pages.
    for item in book.get_items():
        # Process only EpubHtml items
        if isinstance(item, epub.EpubHtml):
            # Parse the HTML content using BeautifulSoup to extract text
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Extract full chapter text (join all text nodes, with spaces)
            chapter_text = soup.get_text(separator=' ', strip=True)

            # Attempt to extract a chapter title: search for the first header tag (h1 or h2)
            header = soup.find(['h1', 'h2'])
            if header and header.get_text().strip():
                chapter_title = header.get_text().strip()
            else:
                # In case no header is found, use first 20 chars
                chapter_title = chapter_text[:20]


            chapters.append({
                "title": chapter_title,
                "text": chapter_text
            })
    return chapters

def text_to_speech_chapter(text, voice='af_heart', sample_rate=24000):
    """
    Given the TTS pipeline and text content, iterates over the generator
    produced by the pipeline to gather audio segments and merges them into a single numpy array.
    """
    audio_segments = []
    # The generator yields tuples (gs, ps, audio_segment)
    generator = pipeline(text, voice=voice)
    for i, (gs, ps, audio_segment) in enumerate(generator):
        # Each 'audio_segment' is expected to be a numpy array.
        audio_segments.append(audio_segment)

    # Merge all audio segments (if any) into one final array.
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
    else:
        full_audio = np.array([])
    return full_audio

def main():

    input_epub = 'book.epub'
    output_dir = "./output"

    os.makedirs(output_dir, exist_ok=True)

    chapters = read_epub(input_epub)
    for idx, chapter in enumerate(chapters):
        chapter_title = chapter["title"] if len(chapter["text"]) > 0 else 'Unknown'
        print(idx, chapter_title)

    # Iterate over chapters with a progress bar to show status, rate, and ETA.
    for idx, chapter in enumerate(tqdm(chapters, desc="Processing chapters", unit="chapter")):
        chapter_text = chapter["text"]
        chapter_title = chapter["title"] if len(chapter["text"]) > 0 else 'Unknown'

        if (len(chapter_text) >= 50):

            start_time = time.time()
            # Generate the complete audio for this chapter using TTS.
            full_audio = text_to_speech_chapter(chapter_text)
            elapsed = time.time() - start_time

            safe_title = re.sub(r'[^\w\s-]', '', chapter_title).strip().replace(' ', '_')
            output_filename = os.path.join(output_dir, f"{idx+1}-{safe_title}.wav")

            # Write the merged audio data to a .wav file.
            sf.write(output_filename, full_audio, 24000)

            # Optionally, print status update for the chapter.
            tqdm.write(f"Chapter {idx+1}: '{chapter_title}' processed in {elapsed:.2f}s; saved as: {output_filename}")


if __name__ == "__main__":
    main()
