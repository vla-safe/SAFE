import time

class Timer:
    def __init__(self, label: str = "Block"):
        """
        label: An optional label to identify the code block in the output.
        """
        self.label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"{self.label} took {elapsed:.6f} seconds.")