from collections import OrderedDict
from gooey import Gooey, GooeyParser
from pathlib import Path
import datetime

from model.Ridge.build import (
        RidgeBuildConfig,
        RidgeBuildConfigList,
        RidgeBuild,
        )
from model.Ridge.generator import (
        RidgeGeneratorConfig,
        RidgeGeneratorConfigList, 
        RidgeGenerator,
        )
from model.Ridge.train_config import (
        RidgeTrainConfig,
        RidgeTrain
        )
from model.Ridge.test_config import RidgeTestConfig, RidgeTest

from model.Ridge.config_samples \
        import (
                #  RidgeTrainConfig,
                RidgeBostonHousePricesTrainConfig)

import pickle


class Run():
    def __init__(self,
                 train_config_list=[
                     RidgeBostonHousePricesTrainConfig(),
                     RidgeTrainConfig(),
                     ],
                 test_config_list=[RidgeTestConfig(),
                                   ],
                 ):
        self.train_config_list = train_config_list
        self.test_config_list = test_config_list

    def _parser(self,
                parser=GooeyParser()
                ):
        subs = parser.add_subparsers()
        self._sub_parser(subs)
        return parser

    def _sub_parser(self,
                    subs=GooeyParser().add_subparsers()
                    ):
        for config in (self.train_config_list):
            parser = subs.add_parser(config.TRAIN_NAME)
            config._parser(parser)

        for config in (self.test_config_list):
            parser = subs.add_parser(config.TEST_NAME)
            config._parser(parser)


    def run(self, config):
        (build_cmds, build_args,
         run_cmds, run_args,
         generator_cmds, generator_args,
         stream) = config
        build_cmd = build_cmds[0]
        run_cmd = run_cmds[0]
        generator_cmd = generator_cmds[0]

        #  stream.put(('Building...', None, None))
        build_config = RidgeBuildConfigList().config(build_cmd, build_args)
        #  build_config = RidgeBuildConfig()
        #  build_config.update(build_args)
        now = datetime.datetime.now()
        log_dir = str(Path(build_config.LOG_DIR).joinpath(
            "{}{:%Y%m%dT%H%M}".format(
                build_config.NAME.lower(), now)))
        model = RidgeBuild(build_config).build()

        stream.put(('Generating...', None, None))
        generator_config = RidgeGeneratorConfigList().config(
                generator_cmd, generator_args)
        #  generator_config = RidgeGeneratorConfig()
        #  generator_config.update(generator_args)
        train_generator, valid_generator = \
            RidgeGenerator(generator_config).train_valid_generator()

        for run_config in self.train_config_list:
            if run_config.TRAIN_NAME == run_cmd:
                run_config.update(run_args)
                model = RidgeTrain(run_config).train(
                        model, train_generator, valid_generator, stream)
                if not Path(log_dir).exists():
                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                filename = Path(log_dir).joinpath('svc_model.sav')
                pickle.dump(model, open(filename, 'wb'))

        for run_config in self.test_config_list:
            if run_config.TEST_NAME == run_cmd:
                run_config.update(run_args)
                return RidgeTest(run_config).test(
                        model, valid_generator, stream)


run_parser = Run()._parser

run = Run().run
