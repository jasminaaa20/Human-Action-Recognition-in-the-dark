# Human Action Recognition in Challenging Lighting Conditions

This project is developed as part of the course CS3501 - Data Science & Engineering Project and is focused on detecting human actions in challenging lighting conditions.

## Table of Contents

- [Introduction](#introduction)
- [Frontend](#frontend)
- [Backend](#backend)
- [Installation](#installation)
- [Important Note](#important-note)

## Introduction

The Human Action Recognition in Challenging Lighting Conditions project is designed to address the detection of human actions when faced with difficult lighting conditions. The project uses a combination of frontend and backend technologies to achieve its goals.

## Client Side

The frontend of the project is built using Next.js, a popular JavaScript framework. To run the frontend, follow these steps:

1. Navigate to the `web_interface/frontend` directory.
2. Run `npm install` to install all frontend dependencies.
3. Start the development server with `npm run dev`.

## Server Side

The backend of the project is powered by Flask, a Python web framework. To run the backend, follow these steps:

1. Create a virtual environment to isolate dependencies. You can use the following command to create a virtual environment:

   ```shell
   python -m venv venv
   ```

2. Activate the virtual environment. Use the appropriate command for your operating system:

   - On Windows:

     ```shell
     venv\Scripts\activate
     ```

3. Install the required Python dependencies using the `requirements.txt` file:

   ```shell
   pip install -r requirements.txt

   ```

4. Navigate to the `web_interface/API` directory.

5. Run the Flask application with `python app.py`.

### Important Note

If you encounter issues with the `dlib` package when installing dependencies, please modify the `requirements.txt` file. You may need to specify the correct path to the `dlib` package. An example line from the `requirements.txt` file might look like this:

`dlib @ file:///F://Project%20Files/dlib-19.24.1-cp311-cp311-win_amd64.whl#sha256=6f1a5ee167975d7952b28e0ce4495f1d9a77644761cf5720fb66d7c6188ae496`
