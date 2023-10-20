from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys
import subprocess

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        # Restart the Flask server when a file is modified
        subprocess.run(['python', 'app.py'])

if __name__ == "__main__":
    path = '.'
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()
    observer.join()