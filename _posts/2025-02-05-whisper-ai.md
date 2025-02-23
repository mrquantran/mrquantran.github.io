---
layout: distill
title: An Introduction to Whisper Architecture with Subtitle Generator
date: 2025-02-05
description: Whisper is a multitask, multilingual model trained on 680,000 hours of diverse audio data. Its transformer-based architecture transcribes, translates, and identifies languages directly from raw audio—all in a single end-to-end pipeline.
tags: speech-to-text
categories: simple-project
thumbnail: assets/img/posts/openai_whisper/approach.png
disqus_comments: true
toc:
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Core Idea
  - name: Architecture
  - name: Inference
  - name: Simple Usage Example
---

Speech-to-text technology has advanced significantly in recent years, with OpenAI's Whisper leading the way. Whisper is an automatic speech recognition (ASR) system that utilizes advanced machine learning methods and a straightforward design to transcribe and translate speech in 99 languages. In this blog, I'll break down how Whisper works, step by step, and demonstrate how it can be used in simple Subtitle Generator projects.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/openai_whisper/model.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="caption">
        Summary of their method: A sequence-to-sequence model, the Transformer model, is trained on various speech processing tasks, such as multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are represented as a sequence of tokens for the decoder to predict, enabling a single model to replace multiple stages of a traditional speech processing pipeline. The multitask training format employs specific tokens that act as task identifiers or classification targets.
    </div>
</div>


## Core Idea
At its core, Whisper is an **encoder-decoder transformer model** trained on 680,000 hours of diverse audio data. Unlike traditional Automatic Speech Recognition (ASR) systems that rely on complex pipelines with separate components for acoustic modeling and language processing, Whisper unifies these tasks into a single neural network. This approach allows it to handle accents, background noise, and multilingual inputs more robustly.

The model’s versatility comes from its ability to perform multiple tasks—like transcription, translation, and language identification—using a unified architecture. By conditioning the decoder with special tokens (e.g., `<|en|>` for English or `<|translate|>` for translation), Whisper dynamically adapts to the user’s needs.

---
## Architecture: From Sound Waves to Text

### Step 1: Processing Audio into Features
Whisper begins by converting raw audio into a **log-Mel spectrogram**, a mathematical representation that captures frequency patterns over time. It transforms chaotic sound waves into a structured format the model can analyze. This spectrogram is generated by:
1. **Resampling**: Audio is standardized to 16,000 Hz.
2. **Spectrogram Creation**: Splitting the waveform into 25-millisecond windows with a 10-millisecond stride, then converting these windows into 80 Mel-frequency bins (mimicking human hearing sensitivity).
1. **Normalization**: The spectrogram is scaled to a range of `[-1, 1]` to stabilize training.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/openai_whisper/mel_spectrogram.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Step 2: The Encoder – Understanding Speech
The encoder outputs a high-dimensional representation of the audio, capturing both acoustic and linguistic features (good blog talking about it [link](https://gattanasio.cc/post/whisper-encoder/)).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/openai_whisper/encoder_whisper.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

#### Convolutional Layers
The spectrogram first passes through two 1D convolutional layers (kernel size 3), which act as a "stem" to detect **local acoustic patterns** like phonemes or formants. These layers:
- Reduce dimensionality while preserving temporal resolution.
- Use GELU activation to introduce non-linearity, enabling the model to learn complex relationships in the audio.

#### Positional Embeddings
Since audio is a time-series, Whisper adds **sinusoidal positional embeddings** to the spectrogram. These embeddings encode the position of each 40ms audio frame, allowing the model to understand temporal order—crucial for distinguishing sequences like "cat" vs. "act".

#### Transformer Blocks
The core of the encoder is a stack of **32 transformer blocks** (for large models). Each block includes:
- **Multi-head self-attention:** Identifies relationships between distant audio segments. For example, it links pronouns ("he") to their later references ("ran") even if separated by seconds of speech.
- **Feed-forward networks:** Refine features using learned transformations.
- **Pre-activation layer normalization:** Stabilizes training by normalizing inputs before attention and feed-forward operations.

Through these layers, the encoder gradually builds a high-dimensional representation (1,500 tokens of 1,280 dimensions for 30-second audio) that encapsulates both acoustic details (e.g., pitch) and linguistic context

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/openai_whisper/encoder_block.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

### Step 3: The Decoder – Generating Text

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/openai_whisper/decoder.png" zoomable=true class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The decoder generates text **one token at a time** using an autoregressive process:

The decoder predicts text tokens autoregressively, one word at a time, using:
1. **Task-Specific Tokens:** Instructions like `<|transcribe|>` or `<|translate|>` guide the decoder49.
2. **Cross-Attention:** The decoder focuses on relevant parts of the encoder’s output while generating each token.
3. **Language Modeling:** Whisper’s training on vast text data helps it produce grammatically correct sentences, even in noisy environments1036.

For example, when translating French to English, the decoder might process:
`<|startoftranscript|><|fr|><|translate|>` → "Bonjour" → "Hello"

---
## Inference
1. **Chunking:** Long audio is split into 30-second segments (or padded to fit).
2. **Batch Processing:** Multiple segments are processed in parallel for efficiency.
3. **Timestamp Prediction:** Optional timestamps mark when each word was spoken.

This pipeline ensures Whisper handles everything from podcasts to phone calls seamlessly.

---
## Simple Usage Example
Using Whisper is straightforward with Open AI whisper Library

### Installation and Setup

To use Whisper, install it via pip:

{% highlight bash %}
pip install openai-whisper
{% endhighlight %}

### Basic Transcription

Here’s how to transcribe an audio file in Python:

{% highlight python %}
import whisper
# Load the model (choose 'tiny', 'base', 'small', 'medium', or 'large')
model = whisper.load_model("base")
# Transcribe audio
result = model.transcribe("audio.mp3")
print(result["text"])
{% endhighlight %}

- **Translation**: Use `model.transcribe(..., task="translate")` to convert non-English speech to English text.
- **Timestamps**: Enable word-level timestamps with `model.transcribe(..., verbose=True)`.

### Subtitle Generator Example
You can reference the link on [Kaggle](https://www.kaggle.com/code/ma3ple/transcript-audio-from-playlist-or-single-video)

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/subtitle-generator.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/subtitle-generator.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}
<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}