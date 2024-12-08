# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false


# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget tar xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Download and install ffmpeg
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xJf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    ffmpeg -version && \
    rm -rf ffmpeg-*-amd64-static ffmpeg-release-amd64-static.tar.xz

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/.

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Define the default command to run the application
CMD ["gunicorn", "app:app", "--workers=2", "--threads=4", "--bind=0.0.0.0:8000", "--timeout=100"]
