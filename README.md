# English to French Seq2Seq Translation using TensorFlow

This project implements a sequence-to-sequence (seq2seq) model using TensorFlow to translate English sentences to French sentences. Seq2seq models are widely used in machine translation tasks, and this project serves as an example of how to build and train such a model for language translation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Overview

The main goal of this project is to demonstrate the implementation of a seq2seq model for translating English sentences to French sentences. Sequence-to-sequence models consist of an encoder and a decoder, where the encoder processes the input sequence (English sentence) and produces a context vector, which the decoder then uses to generate the output sequence (French sentence).

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:

```bash
pip install tensorflow
pip install keras
```
## Usage
1. Just run neural machine translation.ipynb file to get .h5 file of model.
2. Now run nmt_app.py 
```bash
streamlit run nmt_app.py
```
## Examples
I have shown some snaps of examples how you can apply it on some english sentences which can be shown in folder named tested sentences.

## License
This project is licensed under the **MIT License**.
