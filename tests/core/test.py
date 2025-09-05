import hydra
import omegaconf

class B:
    def __init__(self):
        pass

class A:
    def __init__(self, member_cfg):
        print(member_cfg)
        self.B = hydra.utils.instantiate(member_cfg)


cfg_dict = {
        "_target_": "tests.core.test.A",
        "_recursive_": False,
        "member_cfg": {
            "_target_": "tests.core.test.B",
        },
    }

cfg = omegaconf.OmegaConf.create(cfg_dict)
a = hydra.utils.instantiate(cfg)
print(a)
print(a.B)