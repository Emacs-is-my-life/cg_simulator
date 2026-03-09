import queue
import time
import threading
from pathlib import Path
import orjson


class Logger:
    """
    Trace collects traces from modules(core, compute, mm, disk, ...),
    then periodically flush them to disk.
    """

    def __init__(self, output_path: str = "simulation_result.json", flush_period: float = 0.5):
        self.trace_queue = queue.Queue()
        self.output_path = Path(output_path)
        self.flush_period = flush_period
        self.file_ptr = None

        self.stop_event = threading.Event()
        self.worker = None

    def _open_file(self):
        self.file_ptr = open(self.output_path, "w")
        self.file_ptr.write('{"traceEvents": [\n')
        self.file_ptr.flush()

    def _close_file(self):
        self.file_ptr.flush()
        # Take care of the last trailing ",\n"
        pos = self.file_ptr.tell()
        read_back = 2
        self.file_ptr.seek(max(pos - read_back, 0))
        tail = self.file_ptr.read(read_back)

        if tail == ",\n":
            self.file_ptr.seek(max(pos - read_back, 0))
            self.file_ptr.truncate()

        # Append the JSON structure closing, then close the file
        self.file_ptr.write("\n]}\n")
        self.file_ptr.close()

    def _flush(self):
        """Drain the trace_queue and writes events to the json file on disk."""
        arr_trace_event = []
        while True:
            try:
                trace_event = self.trace_queue.get_nowait()
                arr_trace_event.append(trace_event)
            except queue.Empty:
                break

        if arr_trace_event:
            # Write each trace event to the file
            text_to_write = ""
            for trace_event in arr_trace_event:
                trace_text = orjson.dumps(trace_event).decode() + ",\n"
                text_to_write += trace_text

            self.file_ptr.write(text_to_write)
            self.file_ptr.flush()

    def _run(self):
        """
        Internal loop run by the trace thread, periodically flushing trace events to the disk.
        """
        try:
            while not self.stop_event.is_set():
                self._flush()
                time.sleep(self.flush_period)
            # Final one flush when the stop signal is delivered
            self._flush()
        except Exception as e:
            # Simulator exploded somehow
            print(f"Trace module encountered an exception: {e}")
            self._flush()
        finally:
            self._close_file()

    def start(self):
        """
        Starts the trace module, running in a separate thread
        """
        self._open_file()
        self.worker = threading.Thread(target=self._run, name="TraceFlusher")
        self.worker.start()

    def stop(self):
        """
        Signals the trace thread to finish, flush all remaining events,
        then close the json file.
        """
        self.stop_event.set()
        self.worker.join()

    def record(self, trace_event: dict):
        """
        Other modules call this method to record a trace event.
        This method is thread-safe.
        """
        self.trace_queue.put(trace_event)
