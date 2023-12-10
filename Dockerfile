# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app/
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /usr/src/app/
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run test.py when the container launches
CMD ["python", "./RE_Predictor.py"]