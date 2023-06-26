# WangLab Chat Project

This project aims to create a question-answering chatbot using the langchain package. The chatbot is deployed as a FastAPI application and hosted on Render.


## FastAPI Application

The FastAPI application is defined in a script named `app.py`. It uses a language model to generate responses to chat prompts. The model is trained on a dataset scraped from a website and processed into smaller chunks. The chat model is then used to create a question-answering chain, which generates responses for the chat queries.

The application exposes a single endpoint, `/chain`, which accepts POST requests. Each request should have a JSON body with a `message` and `chat_history`. The endpoint responds with the generated chat response.

## Deployment on Render

To deploy the FastAPI application on Render, follow these steps:

1. Create a GitHub repository and add the FastAPI script (`app.py`), the `requirements.txt` file, and the dataset.
2. Set up a new Web Service on Render, linked to your GitHub repository.
3. Set the environment to Python 3.
4. Set the "Build Command" to `pip install -r requirements.txt` and the "Start Command" to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Set the following environment variables:
    - Key: `PYTHON_VERSION`, Value: `3.10.0`
    - Key: `OPENAI_API_KEY`, Value: [your open_api_key]

Render will automatically build and deploy your application, and it will remain synced with your GitHub repository.

## Testing the Application

You can test the application by sending POST requests to the `/chain` endpoint. Here's an example Python script to test the application:

```python
import requests

data = {
    "message": "what is your name?",
    "chat_history": ""
}

response = requests.post("[your render link]/chain", json=data)
print(response.text)
```

To continue the conversation, update the `message` and `chat_history` in the `data` dictionary and send another request.

## Further Details

For more detailed information about the model building and data processing steps, please refer to the Jupyter notebook provided in the repository. The notebook includes code, explanations, and examples to help you understand the inner workings of the project.
