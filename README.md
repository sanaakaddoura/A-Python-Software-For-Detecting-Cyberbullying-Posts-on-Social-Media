# A-Python-Software-For-Detecting-Cyberbullying-Posts-on-Social-Media
This software implements a reproducible pipeline for Arabic text classification, specifically cyberbullying detection, using fine-tuned pre-trained transformer-based language model. The software is designed for research purposes and real time implementation. 

# Project Overview

The project title is "A Python Software For Detecting Cyberbullying Posts on Social Media".

The data used in this research is publicly available at [ArCyC](https://data.mendeley.com/datasets/z2dfgrzx47/1)

## Project Installation and Setup
This guide will walk you through the installation and setup process for the project, which focuses on detecting Arabic cyberbullying content posted on social media using a domain-specific pre-trained language model.

__Prerequisites__  
Before getting started, ensure that you have the following dependencies installed:  

Python 3.x  
pip package manager  

__Dependencies__  
The project requires the following libraries to be installed:  

- transformers
- pyarabic
- arabert
- torch
- emoji
- Numpy
- Pandas
- re
- string
- sentencepiece 

To install the dependencies, run the following command:  
_pip install transformers pyarabic arabert torch emoji numpy pandas re string sentencepiece_  

__Installation Steps__  
Follow the steps below to install and set up the project:  

Clone the repository from codeocean:
_git clone https://git.codeocean.com/capsule-xxx.git_  
  
Navigate to the project directory:  
_cd your-project_  
  
Create a virtual environment (optional but recommended):  
_python -m venv env_  
  
Activate the virtual environment:  
For Windows: _env\Scripts\activate_  
  
For macOS and Linux: _source env/bin/activate_  


## Usage

Run the main script to start disambiguating words:
_python main.py_

## Fine-tuning:
In addition to testing on cyberbullying data, you can finetune the model saved (https://huggingface.co/hugsanaa/CyberAraBERT) on your own dataset and then test the model.

## Configuration

No specific configuration is required for this project.

## Contributing

We welcome contributions from the community! To contribute to this project, please follow the guidelines below:

- Clone the repository and create your branch.
- Make your changes and test them thoroughly.
- Submit a pull request clearly explaining the changes you've made.

Please adhere to our code style conventions and ensure your code is well-documented.

## License

This project is licensed under the Apache License 2.0. For more details, see the [License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) file.

## Support and Contact

If you have any questions, issues, or suggestions, please feel free to open an issue at (https://huggingface.co/hugsanaa/CyberAraBERT) or contact us at sanaa.kaddoura@zu.ac.ae  

