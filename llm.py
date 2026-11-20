from google import genai
from dotenv import load_dotenv
import os
from google.genai.errors import ServerError, ClientError
import time

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

class RetryExhaustedError(Exception):
    def __init__(self, attempts):
        super().__init__(f"gave up after {attempts} attempts")
        self.attempts = attempts

def with_retry(fn, max_retries=3, base_delay=1):
    for attempt in range(max_retries+1):
        try:
            return fn()
        except ServerError as e:
            print(f"ServerError caught on attempt {attempt + 1}")
            if attempt == max_retries:
                raise RetryExhaustedError(attempt+1) from e
            wait_time = 2 ** attempt + base_delay
            print(f"Failed attempt {attempt+1}, waiting {wait_time}s")
            time.sleep(wait_time)
        except ClientError as e:
            print(f"ClientError caught on attempt {attempt + 1}, code={e.code}")
            if e.code != 429:
                raise
            if attempt == max_retries:
                raise RetryExhaustedError(attempt+1) from e
            wait_time = 2 ** attempt + base_delay
            print(f"Failed attempt {attempt+1}, waiting {wait_time}s")
            time.sleep(wait_time)
            
