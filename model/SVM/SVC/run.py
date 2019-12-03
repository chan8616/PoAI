from collections import OrderedDict
from gooey import Gooey, GooeyParser
from pathlib import Path

from model.SVM.SVC.build import (
        SVCBuildConfig,
        SVCBuildConfigList,
        SVCBuild,
        )
from model.SVM.SVC.generator import (
        SVCGeneratorConfig,
        SVCGeneratorConfigList, 
        SVCGenerator,
        )
from model.SVM.SVC.train_config import (
        SVCTrainConfig,
        SVCTrain
        )
from model.SVM.SVC.test_config import SVCTestConfig, SVCTest

from model.SVM.SVC.config_samples \
        import (
                #  SVCTrainConfig,
                SVCIRISTrainConfig)

import pickle


class Run():
    def __init__(self,
                 train_config_list=[
                     SVCIRISTrainConfig(),
                     SVCTrainConfig(),
                     ],
                 test_config_list=[SVCTestConfig(),
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
        build_config = SVCBuildConfigList().config(build_cmd, build_args)
        #  build_config = SVCBuildConfig()
        #  build_config.update(build_args)
        model = SVCBuild(build_config).build()

        stream.put(('Generating...', None, None))
        generator_config = SVCGeneratorConfigList().config(
                generator_cmd, generator_args)
        #  generator_config = SVCGeneratorConfig()
        #  generator_config.update(generator_args)
        train_generator, valid_generator = \
            SVCGenerator(generator_config).train_valid_generator()

        for run_config in self.train_config_list:
            if run_config.TRAIN_NAME == run_cmd:
                run_config.update(run_args)
                model = SVCTrain(run_config).train(
                        model, train_generator, valid_generator, stream)

                Path(build_config.LOG_DIR).mkdir(parents=True, exist_ok=True)
                filename = Path(build_config.LOG_DIR).joinpath(
                        'svc_model.sav')
                pickle.dump(model, open(filename, 'wb'))

        for run_config in self.test_config_list:
            if run_config.TEST_NAME == run_cmd:
                run_config.update(run_args)
                return SVCTest(run_config).test(
                        model, valid_generator, stream)


run_parser = Run()._parser

run = Run().run
