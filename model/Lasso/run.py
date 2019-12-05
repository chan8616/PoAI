from collections import OrderedDict
from gooey import Gooey, GooeyParser
from pathlib import Path
import datetime

from model.Lasso.build import (
        LassoBuildConfig,
        LassoBuildConfigList,
        LassoBuild,
        )
from model.Lasso.generator import (
        LassoGeneratorConfig,
        LassoGeneratorConfigList, 
        LassoGenerator,
        )
from model.Lasso.train_config import (
        LassoTrainConfig,
        LassoTrain
        )
from model.Lasso.test_config import LassoTestConfig, LassoTest

from model.Lasso.config_samples \
        import (
                #  LassoTrainConfig,
                LassoBostonHousePricesTrainConfig)

import pickle


class Run():
    def __init__(self,
                 train_config_list=[
                     LassoBostonHousePricesTrainConfig(),
                     LassoTrainConfig(),
                     ],
                 test_config_list=[LassoTestConfig(),
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
        build_config = LassoBuildConfigList().config(build_cmd, build_args)
        #  build_config = LassoBuildConfig()
        #  build_config.update(build_args)
        now = datetime.datetime.now()
        log_dir = str(Path(build_config.LOG_DIR).joinpath(
            "{}{:%Y%m%dT%H%M}".format(
                build_config.NAME.lower(), now)))
        model = LassoBuild(build_config).build()

        stream.put(('Generating...', None, None))
        generator_config = LassoGeneratorConfigList().config(
                generator_cmd, generator_args)
        #  generator_config = LassoGeneratorConfig()
        #  generator_config.update(generator_args)
        train_generator, valid_generator = \
            LassoGenerator(generator_config).train_valid_generator()

        for run_config in self.train_config_list:
            if run_config.TRAIN_NAME == run_cmd:
                run_config.update(run_args)
                model = LassoTrain(run_config).train(
                        model, train_generator, valid_generator, stream)
                if not Path(log_dir).exists():
                    Path(log_dir).mkdir(parents=True, exist_ok=True)
                filename = Path(log_dir).joinpath('svc_model.sav')
                pickle.dump(model, open(filename, 'wb'))

        for run_config in self.test_config_list:
            if run_config.TEST_NAME == run_cmd:
                run_config.update(run_args)
                return LassoTest(run_config).test(
                        model, valid_generator, stream)


run_parser = Run()._parser

run = Run().run
