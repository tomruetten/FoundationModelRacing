import unittest

import numpy as np

from racing.lidar import (
    RaySamples, compose_ray_samples, generate_rays, sample_cubemap,
    triangle_ray_samples,
)


def samples(ranges, probability=1, intensity=.5):
    r = np.asarray(ranges, dtype=np.float32)
    return RaySamples(r, np.isfinite(r), np.full(r.shape, intensity),
                      np.full(r.shape, probability))


class LidarTests(unittest.TestCase):
    def test_zero_return_car_occludes_background(self):
        result = compose_ray_samples(samples([8]), samples([2], probability=0), seed=1)
        np.testing.assert_equal(result.ranges, [2])
        np.testing.assert_equal(result.hit_mask, [False])
        np.testing.assert_equal(result.return_probability, [0])

    def test_barrier_hides_car_and_nearest_actor_wins(self):
        result = compose_ray_samples(samples([2, 8, np.inf]),
                                     [samples([4, 6, 7]), samples([5, 3, 4])], seed=2)
        np.testing.assert_equal(result.ranges, [2, 3, 4])
        self.assertTrue(result.hit_mask.all())

    def test_seeded_returns_reproducible_and_keep_per_ray_geometry(self):
        static = samples(np.arange(1, 100), probability=.5)
        a = compose_ray_samples(static, [], seed=42)
        b = compose_ray_samples(static, [], seed=42)
        np.testing.assert_equal(a.hit_mask, b.hit_mask)
        np.testing.assert_equal(a.ranges, static.ranges)
        self.assertGreater(a.hit_mask.sum(), 20)
        self.assertLess(a.hit_mask.sum(), 80)

    def test_cube_axial_depth_becomes_range_and_seams_are_assigned_once(self):
        rays = np.array([[1, 0, 0], [1, 1, 0], [1, -1, 0],
                         [0, 0, 1], [-1, -1, -1]], dtype=float)
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        # A cube centered on the sensor has plane depth 2 on every face.
        depth = np.full((6, 8, 8), 2.)
        alpha = np.ones_like(depth)
        colors = np.zeros((6, 8, 8, 3))
        colors[..., 0] = np.arange(6)[:, None, None]
        result = sample_cubemap(rays, depth, alpha, colors, return_head='none')
        np.testing.assert_allclose(result.ranges, [2, 2*np.sqrt(2), 2*np.sqrt(2), 2, 2*np.sqrt(3)])
        self.assertTrue(result.hit_mask.all())
        # Tie belongs to the first maximal face, never overwritten by another face.
        np.testing.assert_equal(result.intensity, [0, 0, 0, 5, 1])

    def test_alpha_geometry_is_independent_of_learned_return_head(self):
        d = np.ones((6, 4, 4))
        c = np.zeros((6, 4, 4, 3))
        c[..., 1] = np.log(3)
        rays = np.array([[1., 0, 0]])
        for head, expected in [('none', 1), ('return', .75), ('drop', .25)]:
            result = sample_cubemap(rays, d, d, c, return_head=head)
            self.assertTrue(result.hit_mask[0])
            self.assertAlmostEqual(result.return_probability[0], expected)
        result = sample_cubemap(rays, d, d*.5, c)
        self.assertFalse(result.hit_mask[0])

    def test_triangle_raycast_plane_known_ranges_and_sensor_pose(self):
        triangles = np.array([[[3, -10, -10], [3, 10, -10], [3, 0, 10]]])
        pose = np.eye(4)
        pose[0, 3] = 1
        rays = np.array([[1, 0, 0], [1, .5, 0], [-1, 0, 0]], dtype=float)
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        result = triangle_ray_samples(pose, rays, triangles, intensity=.7)
        np.testing.assert_allclose(result.ranges[:2], [2, np.sqrt(5)])
        self.assertFalse(result.hit_mask[2])
        self.assertAlmostEqual(result.intensity[0], .7, places=6)

    def test_generator_is_unit_length_and_rejects_invalid_input(self):
        rays = generate_rays(200, 3)
        self.assertEqual(rays.shape, (200, 3))
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-6)
        with self.assertRaises(ValueError):
            sample_cubemap(np.zeros((1, 3)), np.ones((6, 4, 4)),
                           np.ones((6, 4, 4)), np.zeros((6, 4, 4, 3)))


if __name__ == '__main__':
    unittest.main()
