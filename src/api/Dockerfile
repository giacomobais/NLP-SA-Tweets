# 1. Start with a base image
FROM huggingface/transformers-pytorch-gpu:latest

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements.txt from src/api to the working directory
COPY src/api/requirements.txt . 

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy everything from the project root into /app in the container
COPY . .

# 6. Expose the port (optional, adjust as needed)
EXPOSE 8000

# 7. Specify the command to run the web app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 
