FROM python:3.11-slim

# Install system dependencies for PyTorch and Hugging Face
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements.txt from src/api to the working directory
COPY src/api/requirements.txt . 

# 4. Install dependencies
RUN pip install -r requirements.txt

# 5. Copy everything from the project root into /app in the container
COPY . .

# 6. Expose the port (optional, adjust as needed)
EXPOSE 8000

# 7. Specify the command to run the web app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 
