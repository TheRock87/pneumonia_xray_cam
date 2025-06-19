# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
COPY ./app /code/app
COPY ./model /code/model

# Command to run the app. Uvicorn is the server.
# The host 0.0.0.0 makes it accessible from outside the container.
# Hugging Face Spaces expects the app to run on port 7860.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
