# Image Description with LLaVA Model and Translation to Arabic

This project demonstrates how to use the LLaVA model for generating image descriptions and translating to Arabic with Google translation. The application processes an image, generates a textual description in English, and translates it to Arabic. The project involves loading an image, processing it, creating prompts for the model, generating a response, and translating the response.

## Features

- **Image Processing**: Load and preprocess images for model inference.
- **Prompt Creation**: Generate prompts for the LLaVA model.
- **Image Description**: Use the LLaVA model to describe images.
- **Translation**: Translate the generated description from English to Arabic.
- **Markdown Conversion**: Convert the translated text to Markdown format for better presentation.

## Prerequisites

To run this project, you need the following:

- Python 3.7 or later
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/MahmoudIbrahims/llava-image-description.git
    cd llava-image-description
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have an image file named `bike-girl.jpeg` or replace the image file path in the script with your own image.

2. Run the script:

    ```python
    from model import llava_model
    from preprocessing import load_image, process_image
    from preprocessing import create_prompt, ask_image, to_markdown
    from translation import transArabic, transEnglish

    tokenizer, model, image_processor, context_len = llava_model()

    image = load_image("bike-girl.jpeg")

    processed_image = process_image(image, image_processor, model)
    print(type(processed_image), processed_image.shape)

    prompt, _ = create_prompt("Describe the image")
    print(prompt)

    response = ask_image(image, "Describe the image", tokenizer, model)

    result = transArabic(response)

    to_markdown(result)
    ```

3. The script will process the image, generate a description, translate it to Arabic, and convert it to Markdown format.

## Code Overview

### `model.py`

Contains the function `llava_model()` to load the LLaVA model and its components.

### `preprocessing.py`

Contains functions for image loading, processing, prompt creation, generating descriptions, and converting text to Markdown.

- **`load_image(filepath)`**: Loads an image from the given filepath.
- **`process_image(image, image_processor, model)`**: Processes the image using the image processor and model.
- **`create_prompt(description)`**: Creates a prompt for the model.
- **`ask_image(image, description, tokenizer, model)`**: Generates a description for the image using the model.
- **`to_markdown(text)`**: Converts text to Markdown format.

### `translation.py`

Contains functions for translating text between English and Arabic.

- **`transArabic(text)`**: Translates English text to Arabic.
- **`transEnglish(text)`**: Translates Arabic text to English.


## use
i used torch==2.1 --progress-bar off

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Acknowledgments

Special thanks to the developers of the LLaVA model and the contributors to the various libraries used in this project.










