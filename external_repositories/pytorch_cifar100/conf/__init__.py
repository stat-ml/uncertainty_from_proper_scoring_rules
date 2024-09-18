"""dynamically load settings

author baiyu
"""

import sys

sys.path.insert(0, "./external_repos/pytorch_cifar100")
import conf.global_settings as settings


class Settings:
    def __init__(self, settings):
        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))


settings = Settings(settings)
