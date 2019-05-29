from gooey.python_bindings.config_generator import create_from_parser
from gooey.gui.util.freeze import getResourcePath
from gooey.gui.lang import i18n
from gooey.gui import image_repository
from gooey.util.functional import merge


def build_spec_from_parser(parser, source_path=None, **kwds):
    build_spec = create_from_parser(parser,
                                    source_path=source_path, **kwds)
    build_spec.update(
        {'language_dir': getResourcePath('languages'),
         'image_dir': '::gooey/default'})

    i18n.load(build_spec['language_dir'],
              build_spec['language'],
              build_spec['encoding'])
    imagesPaths = image_repository.loadImages(
        build_spec['image_dir'])
    build_spec = merge(build_spec, imagesPaths)
    return build_spec
