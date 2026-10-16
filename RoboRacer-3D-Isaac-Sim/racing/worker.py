"""Persistent LiDAR subprocess over AF_UNIX control and shared-memory arrays.

Only metadata travels over the socket. The owning client allocates/unlinks
the pose, rays and output buffer; the child attaches for each request. Torch
and gsplat are imported only inside the worker's explicitly selected Python.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np

from .lidar import RaySamples, _pose, _unit_rays


class StaleFrameError(ValueError):
    """A frame belongs to another episode or regresses sequence/simulation time."""


class _FrameOrder:
    def __init__(self):
        self.episode = None
        self.sequence = -1
        self.time_s = -np.inf
        self.retired = set()

    def reset(self, episode):
        if not isinstance(episode, (str, int)) or isinstance(episode, bool):
            raise ValueError('episode_id must be a string or integer')
        if episode == self.episode or episode in self.retired:
            raise StaleFrameError('Reset needs a new episode_id')
        if self.episode is not None:
            self.retired.add(self.episode)
        self.episode, self.sequence, self.time_s = episode, -1, -np.inf

    def validate(self, episode_id, sequence, time_s):
        if not isinstance(episode_id, (str, int)) or isinstance(episode_id, bool):
            raise ValueError('episode_id must be a string or integer')
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError('sequence must be a nonnegative integer')
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('time_s must be finite and nonnegative')
        if ((self.episode is not None and episode_id != self.episode) or
                sequence <= self.sequence or time_s < self.time_s or episode_id in self.retired):
            raise StaleFrameError('Stale episode, sequence, or simulation time')

    def accept(self, episode_id, sequence, time_s):
        self.validate(episode_id, sequence, time_s)
        self.episode, self.sequence, self.time_s = episode_id, sequence, time_s


def _receive(sock):
    def read_exact(n):
        chunks = bytearray()
        while len(chunks) < n:
            data = sock.recv(n-len(chunks))
            if not data:
                raise EOFError('LiDAR worker connection closed')
            chunks.extend(data)
        return chunks
    size, = struct.unpack('!I', read_exact(4))
    if size > 65536:
        raise ValueError('Oversized worker metadata')
    return json.loads(read_exact(size))


def _send(sock, value):
    payload = json.dumps(value, allow_nan=False).encode()
    if len(payload) > 65536:
        raise ValueError('Oversized worker metadata')
    sock.sendall(struct.pack('!I', len(payload)) + payload)


class WorkerClient:
    """Synchronous client; one instance per sensor stream, not thread safe.

    ``diagnostic=True`` selects an explicit constant-range protocol fixture.
    It never activates as a fallback when loading or rendering a model fails.
    """
    def __init__(self, python_executable, checkpoint, *, device='cuda', return_head='none',
                 alpha_threshold=.99, resolution=256, max_range=8., timeout_s=10.,
                 startup_timeout_s=120., diagnostic=False, diagnostic_delay_s=0.):
        self.python_executable = str(python_executable)
        if not diagnostic and checkpoint is None:
            raise ValueError('A checkpoint is required for learned LiDAR')
        checkpoint = str(Path(checkpoint).expanduser().resolve(strict=True)) if checkpoint is not None else None
        if timeout_s <= 0 or startup_timeout_s <= 0 or diagnostic_delay_s < 0:
            raise ValueError('Timeouts must be positive; diagnostic delay must be nonnegative')
        self.config = dict(checkpoint=checkpoint, device=device, return_head=return_head,
                           alpha_threshold=alpha_threshold, resolution=resolution, max_range=max_range,
                           diagnostic=diagnostic, diagnostic_delay_s=diagnostic_delay_s)
        self.timeout_s, self.startup_timeout_s = timeout_s, startup_timeout_s
        self.process = None
        self._socket = self._memory = None
        self._closed = False
        self._order = _FrameOrder()

    @property
    def shared_memory_name(self):
        return None if self._memory is None else self._memory.name

    def start(self):
        if self._closed:
            raise RuntimeError('WorkerClient is closed')
        if self.process is not None:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError('LiDAR worker exited')
            return self
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket = parent
        # Isaac's bundled Python/torch libraries must not leak into roboracer.
        env = os.environ.copy()
        env.pop('PYTHONHOME', None)
        env.pop('LD_LIBRARY_PATH', None)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
        try:
            self.process = subprocess.Popen(
                [self.python_executable, '-m', 'racing.worker', '--fd', str(child.fileno())],
                pass_fds=(child.fileno(),), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            child.close()
            parent.settimeout(self.startup_timeout_s)
            _send(parent, self.config)
            reply = _receive(parent)
            if reply.get('status') != 'ready':
                raise RuntimeError(reply.get('error', 'Worker initialization failed'))
            parent.settimeout(self.timeout_s)
        except BaseException:
            child.close()
            self.close()
            raise
        return self

    def _rpc(self, request):
        try:
            _send(self._socket, request)
            reply = _receive(self._socket)
            if reply.get('status') != 'ok':
                raise RuntimeError(reply.get('error', 'LiDAR worker request failed'))
            for field in ('episode_id', 'sequence', 'time_s'):
                if field in request and reply.get(field) != request[field]:
                    raise StaleFrameError(f'Worker response {field} does not match request')
            return reply
        except BaseException:
            # Timeout or stale output cannot be allowed to contaminate a later
            # frame; terminate before unlinking a buffer the child may still use.
            self.close()
            raise

    def reset(self, episode_id):
        self.start()
        # Validate before changing either side's episode state.
        if episode_id == self._order.episode or episode_id in self._order.retired:
            raise StaleFrameError('Reset needs a new episode_id')
        self._rpc(dict(op='reset', episode_id=episode_id))
        self._order.reset(episode_id)

    def render(self, pose, rays, *, episode_id, sequence, time_s):
        pose, rays = _pose(pose), _unit_rays(rays)
        self._order.validate(episode_id, sequence, time_s)
        self.start()
        n = len(rays)
        size = (16+7*n)*np.dtype(np.float64).itemsize
        if self._memory is None or self._memory.size != size:
            self._release_memory()
            self._memory = shared_memory.SharedMemory(create=True, size=size)
        buffer = np.ndarray((16+7*n,), dtype=np.float64, buffer=self._memory.buf)
        buffer[:16] = pose.ravel()
        buffer[16:16+3*n] = rays.ravel()
        request = dict(op='lidar', episode_id=episode_id, sequence=sequence, time_s=float(time_s),
                       n_rays=n, shared_memory=self._memory.name)
        self._rpc(request)
        output = buffer[16+3*n:].reshape(n, 4).copy()
        self._order.accept(episode_id, sequence, time_s)
        return RaySamples(output[:, 0], output[:, 1].astype(bool), output[:, 2], output[:, 3])

    def _release_memory(self):
        if self._memory is not None:
            self._memory.close()
            self._memory.unlink()
            self._memory = None

    def close(self):
        if self._socket is not None:
            try:
                self._socket.settimeout(.1)
                _send(self._socket, dict(op='close'))
            except (OSError, EOFError):
                pass
            self._socket.close()
            self._socket = None
        if self.process is not None and self.process.poll() is None:
            try:
                # CUDA module teardown can take longer than the diagnostic
                # process. Allow a bounded graceful exit before terminating.
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._release_memory()
        self._closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def _serve(sock):
    config = _receive(sock)
    try:
        diagnostic = config.pop('diagnostic', False)
        delay = config.pop('diagnostic_delay_s', 0.)
        if diagnostic:
            renderer = None
        else:
            from .lidar import LearnedLidar
            renderer = LearnedLidar(**config)
        _send(sock, dict(status='ready'))
    except Exception as exc:
        _send(sock, dict(status='error', error=f'{type(exc).__name__}: {exc}'))
        return
    order = _FrameOrder()
    while True:
        try:
            request = _receive(sock)
        except EOFError:
            return
        if request.get('op') == 'close':
            return
        metadata = {k: request[k] for k in ('episode_id', 'sequence', 'time_s') if k in request}
        try:
            if request.get('op') == 'reset':
                order.reset(request['episode_id'])
            elif request.get('op') == 'lidar':
                order.validate(**metadata)
                n = request['n_rays']
                if not isinstance(n, int) or n < 0:
                    raise ValueError('Invalid ray count')
                memory = shared_memory.SharedMemory(name=request['shared_memory'])
                # The standalone worker has its own tracker. Only the owning
                # parent should unlink this segment (also on Python 3.10).
                from multiprocessing import resource_tracker
                resource_tracker.unregister(memory._name, 'shared_memory')
                try:
                    if memory.size != (16+7*n)*8:
                        raise ValueError('Shared-memory size does not match ray count')
                    buffer = np.ndarray((16+7*n,), dtype=np.float64, buffer=memory.buf)
                    pose = buffer[:16].reshape(4, 4)
                    rays = buffer[16:16+3*n].reshape(n, 3)
                    if diagnostic:
                        time.sleep(delay)
                        result = RaySamples(np.full(n, 3.), np.ones(n, bool), np.full(n, .5), np.ones(n))
                    else:
                        result = renderer.render(pose, rays)
                    output = buffer[16+3*n:].reshape(n, 4)
                    output[:] = np.column_stack((result.ranges, result.hit_mask,
                                                 result.intensity, result.return_probability))
                finally:
                    memory.close()
                order.accept(**metadata)
            else:
                raise ValueError('Unknown worker operation')
            _send(sock, dict(status='ok', **metadata))
        except Exception as exc:
            _send(sock, dict(status='error', error=f'{type(exc).__name__}: {exc}', **metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True, help='Inherited AF_UNIX socket descriptor')
    args = parser.parse_args()
    with socket.socket(fileno=args.fd) as sock:
        _serve(sock)


if __name__ == '__main__':
    main()
