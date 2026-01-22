import os

class Logger:
    path = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(path, "app.log")

    @staticmethod
    def clear_log():
        with open(Logger.LOG_FILE, "w") as log_file:
            log_file.write("")

    @staticmethod
    def log(message: str):
        with open(Logger.LOG_FILE, "a") as log_file:
            log_file.write(f"{message}\n")