## Table of Contents 📑

- [Getting Started](#getting-started) 🚀
   - [Prerequisites](#prerequisites) 📋
   - [Installation](#installation) 🔧
- [Example Usage](#example-usage) 💡
   - [Example Output](#example-output) 🖥️
- [Note](#note) 📌
- [Contributions](#contributions) 🤝
- [License](#license) ⚖️


# Chatlocal 💬
Chatlocal is a Python library designed to interact with your local files and provide AI-powered question-answering capabilities. The library can read your local documents, vectorize them, and then answer questions using the indexed documents. The library uses the FAISS library for efficient similarity search and LangChain for retrieval-based question answering.

Currently, this project is in an alpha stage and is not ready for production use. We are working on improving the library and adding more features. However, you can interact with the library through the command line and ask questions to your indexed documents, currently only .txt or .md files are supported.

## Getting Started 🚀

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites 📋

Chatlocal requires Python 3.9.16 or higher.

### Installation 🔧

We recommend using Poetry for managing Python dependencies. If you do not have Poetry installed, you can install it with the following command on Linux, macOS and Windows (WSL2):

```bash
curl -sSL https://install.python-poetry.org | python -
```

After installing Poetry, navigate to the directory where the `pyproject.toml` file of Chatlocal is located, then install the dependencies with:

```bash
poetry install
```

This will create a virtual environment and install all the necessary dependencies in it.

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

## Example Usage 💡

There are two scripts that showcase the usage of the library:

1. `dev/main.py`: This script loads your documents, vectorizes them, and saves them for later use.
 You can specify the location of your documents and the file types to be processed.

   Run `dev/main.py` with Poetry using the following command:

   ```bash
   poetry run python dev/main.py --docpath [your_document_path] --filetypes [filetypes]
   ```

   Replace `[your_document_path]` with the path to your documents and `[filetypes]` with the
   desired file types (e.g., .md, .txt). If you ommit filetypes, by default this is set to ".md",
    and if you ommit the docpath, by default this is set to "~/code/curriculum".
   this can be changed inside the main.py file.
   Using this on obsidian would look like this:

   ```bash
   poetry run python dev/main.py --docpath=~/code/curriculum
   ```
   assuming that you have your obsidian vault in `~/code/curriculum`.`


2. `qa.py`: This script allows you to interactively ask questions to your indexed documents.

   Run `qa.py` with Poetry using the following command:

   ```bash
   poetry run python qa.py
   ```

   Then, you can start typing your questions. Type `Q` to quit the interactive prompt.

### Example Output 🖥️

```bash
❯ python dev/qa.py
Ask a question: (Q to quit) Kun je lineare regressie uitleggen?

Answer: Linear regression is a statistical approach for modeling the relationship between a scalar
response and one or more explanatory variables. Simple linear regression concerns two-dimensional
sample points with one independent variable and one dependent variable and finds a linear function
that predicts the dependent variable values as a function of the independent variable.
The relationships are modeled using linear predictor functions whose unknown model parameters
are estimated from the data. The law of conditional probability distribution of the response
given the values of the predictors is used.

Sources: /code/curriculum/Nuggets/Linear Regression.md, /code/curriculum/Nuggets/Simple Linear Regression.md
```

```bash
Ask a question: (Q to quit) can you explain linear regression in the style of a haiku?
Answer: Linear regression,
Predictive model for data,
Simple or multiple.

Sources: /code/curriculum/Nuggets/Linear Regression.md, /code/curriculum/Nuggets/Simple Linear Regression.md
```

```bash
Ask a question: (Q to quit) What is kullback leibler?
Answer: Kullback-Leibler Divergence is a measure of difference between two probability distributions
and can be considered as a similarity or distance measure, but it is not a distance in the strict
mathematical sense. It is also known as Relative Entropy or KL Divergence. It is not a distance since
axioms 3 and 4 of the Distance function are not satisfied. It is the difference between Entropy and
Cross Entropy and is a special case of Bregman divergence.

Sources: /code/curriculum/Nuggets/Kullback-Leibler Divergence.md, /code/curriculum/Nuggets/Similarity or Distance Measures.md
```

```bash
Ask a question: (Q to quit) Wat zijn de belangrijkste themas in deze aantekeningen?
Answer: The main themes in these notes are:
- Importance of not throwing away notes
- Need to discuss changes to notes with the owner
- Creating notes in the correct folder to ensure the correct template is added
- Contacting xxxx for questions or missing information
- Information on sharing information and lessons
- Process for copying lecture notes to OnderwijsOnline
- Importance of using learning outcomes as a means rather than an end in testing
- Involving the entire team in the curriculum design process
- Using peer review forms for assessment
- Creating exams from individual questions in Exam Questions

Sources: /code/curriculum/Workflow/Belangrijkste Afspraken.md,
/code/curriculum/Workflow/Werkwijze MADS lessen.md,
/code/curriculum/Toetsing/Review Forms en toetsing.md,
/code/curriculum/Workflow/Overige/Leeruitkomsten formuleren.md
```


## Note 📌
The library stores indexed vectors and document store in the `~/.cache/chatlocal` directory.
This can be changed by setting the `CACHE_DIR` environment variable.
eg
```bash
❯ export CACHE_DIR=/tmp/chatlocal`
```

## Contributions 🤝

Contributions are welcomed. Please create a pull request with your changes.

## License ⚖️

This project is licensed under the MIT License. See the LICENSE file for details.
