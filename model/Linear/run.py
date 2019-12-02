from collections import OrderedDict
from gooey import Gooey, GooeyParser

from model.Linear.build import (
        LinearBuildConfig,
        LinearBuildConfigList,
        LinearBuild,
        )
from model.Linear.generator import (
        LinearGeneratorConfig,
        LinearGeneratorConfigList, 
        LinearGenerator,
        )
from model.Linear.train_config import (
        LinearTrainConfig,
        LinearTrain
        )
from model.Linear.test_config import LinearTestConfig, LinearTest

from model.Linear.config_samples \
        import (
                #  LinearTrainConfig,
                LinearBostonHousePricesTrainConfig)

from keras import backend as K


class Run():
    def __init__(self,
                 train_config_list=[
                     LinearBostonHousePricesTrainConfig(),
                     LinearTrainConfig(),
                     ],
                 test_config_list=[LinearTestConfig(),
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
        K.clear_session()
        build_cmd = build_cmds[0]
        run_cmd = run_cmds[0]
        generator_cmd = generator_cmds[0]

        #  stream.put(('Building...', None, None))
        build_config = LinearBuildConfigList().config(build_cmd, build_args)
        #  build_config = LinearBuildConfig()
        #  build_config.update(build_args)
        model = LinearBuild(build_config).build()

        stream.put(('Generating...', None, None))
        generator_config = LinearGeneratorConfigList().config(
                generator_cmd, generator_args)
        #  generator_config = LinearGeneratorConfig()
        #  generator_config.update(generator_args)
        train_generator, valid_generator = \
            LinearGenerator(generator_config).train_valid_generator()

        for run_config in self.train_config_list:
            if run_config.TRAIN_NAME == run_cmd:
                run_config.update(run_args)
                return LinearTrain(run_config).train(
                        model, train_generator, valid_generator, stream)

        for run_config in self.test_config_list:
            if run_config.TEST_NAME == run_cmd:
                run_config.update(run_args)
                return LinearTest(run_config).test(
                        model, valid_generator, stream)


run_parser = Run()._parser

run = Run().run
