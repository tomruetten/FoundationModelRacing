import sys
import tempfile
import unittest
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import numpy as np

from racing.worker import WorkerClient, StaleFrameError, _receive


class WorkerTests(unittest.TestCase):
    def test_persistent_shared_memory_roundtrip_and_reset(self):
        with WorkerClient(sys.executable, None, diagnostic=True) as client:
            a = client.render(np.eye(4), np.eye(3), episode_id='one', sequence=0, time_s=0)
            process = client.process
            self.assertTrue(a.hit_mask.all())
            np.testing.assert_equal(a.ranges, [3, 3, 3])
            client.render(np.eye(4), np.eye(3), episode_id='one', sequence=1, time_s=.1)
            self.assertIs(client.process, process)
            client.reset('two')
            b = client.render(np.eye(4), np.eye(3)[:1], episode_id='two', sequence=0, time_s=0)
            self.assertEqual(b.ranges.shape, (1,))
            name = client.shared_memory_name
        self.assertIsNotNone(process.poll())
        self.assertEqual(process.returncode, 0)
        with self.assertRaises(FileNotFoundError):
            SharedMemory(name=name)

    def test_stale_episode_sequence_and_time_rejected(self):
        with WorkerClient(sys.executable, None, diagnostic=True) as client:
            args = (np.eye(4), np.eye(3))
            client.render(*args, episode_id='one', sequence=2, time_s=1)
            for episode, seq, t in [('one', 2, 1), ('one', 3, .5), ('two', 3, 2)]:
                with self.assertRaises(StaleFrameError):
                    client.render(*args, episode_id=episode, sequence=seq, time_s=t)
            client.reset('two')
            with self.assertRaises(StaleFrameError):
                client.render(*args, episode_id='one', sequence=4, time_s=2)

    def test_timeout_terminates_worker_and_unlinks_shared_memory(self):
        client = WorkerClient(sys.executable, None, diagnostic=True,
                              diagnostic_delay_s=.5, timeout_s=.1, startup_timeout_s=10)
        client.start()
        process = client.process
        with self.assertRaises(TimeoutError):
            client.render(np.eye(4), np.eye(3), episode_id='one', sequence=0, time_s=0)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(client.shared_memory_name)
        client.close()

    def test_no_implicit_diagnostic_fallback(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            WorkerClient(sys.executable, None)

    def test_stale_response_metadata_closes_stream(self):
        with WorkerClient(sys.executable, None, diagnostic=True) as client:
            process = client.process

            def stale_response(sock):
                response = _receive(sock)
                response['episode_id'] = 'old-episode'
                return response

            with patch('racing.worker._receive', side_effect=stale_response):
                with self.assertRaises(StaleFrameError):
                    client.render(np.eye(4), np.eye(3), episode_id='new', sequence=0, time_s=0)
            self.assertIsNotNone(process.poll())
            self.assertIsNone(client.shared_memory_name)

    def test_worker_init_failure_has_no_diagnostic_fallback(self):
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint:
            checkpoint.write(b'not a checkpoint')
            checkpoint.flush()
            client = WorkerClient(sys.executable, checkpoint.name, device='cpu')
            with self.assertRaises(RuntimeError):
                client.start()
            self.assertIsNotNone(client.process.poll())
            self.assertIsNone(client.shared_memory_name)


if __name__ == '__main__':
    unittest.main()
