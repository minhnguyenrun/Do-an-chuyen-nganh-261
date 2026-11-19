"""Device-wide sampling includes the Paddle child (other processes may contribute)."""
import subprocess
import threading


class DeviceMemory:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.peak = None
        self.error = None
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                                    capture_output=True, text=True, timeout=3,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
            value = float(result.stdout.strip().splitlines()[0])
            self.peak = max(value, self.peak or 0)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.error = str(exc)

    def __enter__(self):
        if self.enabled:
            self.sample()
            def loop():
                while not self.stop.wait(.2):
                    self.sample()
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(4)
            self.sample()

    def report(self):
        return {'peak_used_mb': self.peak, 'error': self.error,
                'scope': 'GPU 0 device-wide nvidia-smi sampling, 200ms interval; includes other processes; may miss short peaks'}
