import numpy as np
import pytest

from racing.assets import _bake_camera_world, _sh_basis
from racing.config import quaternion_matrix


@pytest.mark.parametrize('degree', [0,1,2,3])
def test_world_bake_preserves_directional_radiance_and_covariance(degree):
    rng=np.random.default_rng(13)
    rotation=quaternion_matrix([.3,-.4,.2,.7])
    transform=np.eye(4)
    transform[:3,:3]=rotation*1.7
    transform[:3,3]=[2,-3,.4]
    means=rng.normal(size=(5,3)).astype(np.float32)
    scales=rng.normal(size=(5,3)).astype(np.float32)
    quats=rng.normal(size=(5,4)).astype(np.float32)
    quats/=np.linalg.norm(quats,axis=1,keepdims=True)
    coefficients=rng.normal(size=(5,(degree+1)**2,3)).astype(np.float32)
    result=_bake_camera_world(means,scales,quats,coefficients[:,0],coefficients[:,1:],transform)
    new_means,new_scales,new_quats,new_dc,new_rest=result
    np.testing.assert_allclose(new_means,means@transform[:3,:3].T+transform[:3,3],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(new_scales,scales+np.log(1.7),rtol=1e-6,atol=1e-6)
    for old,new in zip(quats,new_quats):
        expected=rotation@quaternion_matrix(old[[1,2,3,0]])
        np.testing.assert_allclose(quaternion_matrix(new[[1,2,3,0]]),expected,atol=1e-6)
    directions=rng.normal(size=(123,3))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    expected=np.einsum('dk,nkc->ndc',_sh_basis(directions@rotation,degree),coefficients)
    actual=np.einsum('dk,nkc->ndc',_sh_basis(directions,degree),np.concatenate([new_dc[:,None],new_rest],axis=1))
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)


def test_local_sh_basis_matches_gsplat_polynomial_order():
    torch=pytest.importorskip('torch')
    impl=pytest.importorskip('gsplat.cuda._torch_impl')
    rng=np.random.default_rng(22)
    directions=rng.normal(size=(17,3))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    expected=impl._eval_sh_bases_fast(16,torch.from_numpy(directions)).numpy()
    np.testing.assert_allclose(_sh_basis(directions,3),expected,atol=1e-12)
