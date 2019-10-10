"""
Primary orchestration and control point for Gooey.
"""

import sys
from itertools import chain
from copy import deepcopy

import wx
from gui.notebook.build_spec_from_parser import build_spec_from_parser

from gooey.gui import events
from gooey.gui.components.header import FrameHeader
from gooey.gui.components.footer import Footer
from gooey.gui.util import wx_util
from gooey.gui.components.config import ConfigPage, TabbedConfigPage
# from gooey.gui.components.sidebar import Sidebar
from gui.notebook.sidebar import Sidebar
from gooey.gui.components.tabbar import Tabbar
from gooey.util.functional import getin, assoc, flatmap, compact
from gooey.python_bindings import constants
from gooey.gui.pubsub import pub
from gooey.gui import cli
from gooey.gui.components.console import Console
from gooey.gui.lang.i18n import _
from gooey.gui.processor import ProcessController
from gooey.gui.util.wx_util import transactUI
from gooey.gui.components import modals
from gooey.gui import seeder



def buildString(target, cmd, positional, optional):
    return u'{} --ignore-gooey {}'.format(target, cmd_string)


class Page(wx.Panel):
    """
    Main window for Gooey.
    """

    def __init__(self, buildSpec, *args, **kwargs):
        super(Page, self).__init__(*args, **kwargs)
        self._state = {}
        self.buildSpec = buildSpec

        self.header = FrameHeader(self, buildSpec)
        self.configs = self.buildConfigPanels(self)
        self.navbar = self.buildNavigation()
        self.footer = Footer(self, buildSpec)
        self.console = Console(self, buildSpec)
        self.layoutComponent()

        self.clientRunner = ProcessController(
            self.buildSpec.get('progress_regex'),
            self.buildSpec.get('progress_expr'),
            self.buildSpec.get('encoding')
        )

        pub.subscribe(events.WINDOW_START, self.onStart)
        pub.subscribe(events.WINDOW_RESTART, self.onStart)
        pub.subscribe(events.WINDOW_STOP, self.onStopExecution)
        pub.subscribe(events.WINDOW_CLOSE, self.onClose)
        pub.subscribe(events.WINDOW_CANCEL, self.onCancel)
        pub.subscribe(events.WINDOW_EDIT, self.onEdit)
        pub.subscribe(events.CONSOLE_UPDATE, self.console.logOutput)
        pub.subscribe(events.EXECUTION_COMPLETE, self.onComplete)
        pub.subscribe(events.PROGRESS_UPDATE, self.footer.updateProgressBar)
        # Top level wx close event
        self.Bind(wx.EVT_CLOSE, self.onClose)

        if self.buildSpec['poll_external_updates']:
            self.fetchExternalUpdates()

        if self.buildSpec.get('auto_start', False):
            self.onStart()

    def onStart(self, *args, **kwarg):
        """
        Verify user input and kick off the client's program if valid
        """
        with transactUI(self):
            config = self.navbar.getActiveConfig()
            config.resetErrors()
            if config.isValid():
                self.clientRunner.run(self.buildCliString())
                self.showConsole()
            else:
                config.displayErrors()
                self.Layout()

    def onEdit(self):
        """Return the user to the settings screen for further editing"""
        with transactUI(self):
            if self.buildSpec['poll_external_updates']:
                self.fetchExternalUpdates()
            self.showSettings()

    def buildString(self):
        config = self.navbar.getActiveConfig()
        group = self.buildSpec['widgets'][self.navbar.getSelectedGroup()]
        positional = config.getPositionalArgs()
        optional = config.getOptionalArgs()

        positionals = deepcopy(positional)
        if positionals:
            positionals.insert(0, "--")

        cmd_string = compact(chain(optional, positionals))
        print(cmd_string)  # [" ''", ' ', " ''"]
        cmd_string = ["{} ''".format(cmd) if cmd.find("'") == -1 else cmd
                      for cmd in cmd_string]
        print(cmd_string)  # [" ''", " ", " ''"]
        cmd_string = ' '.join(cmd_string)  # " '' ''"
        print(cmd_string)
        cmd_string = cmd_string.split("'")  # [' ', ' ', '']
        print(cmd_string)
        cmd_string = [cmd.strip() for cmd in cmd_string
                      if cmd.strip() != '']  # [' ', ' ']
        print(cmd_string)
        cmd_string = [group['command']] + cmd_string
        print(cmd_string)
        return cmd_string

    def buildCliString(self):
        """
        Collect all of the required information from the config screen and
        build a CLI string which can be used to invoke the client program
        """
        config = self.navbar.getActiveConfig()
        group = self.buildSpec['widgets'][self.navbar.getSelectedGroup()]
        positional = config.getPositionalArgs()
        optional = config.getOptionalArgs()
        print(cli.buildCliString(
            self.buildSpec['target'],
            group['command'],
            positional,
            optional
        ))
        return cli.buildCliString(
            self.buildSpec['target'],
            group['command'],
            positional,
            optional
        )


    def onComplete(self, *args, **kwargs):
        """
        Display the appropriate screen based on the success/fail of the
        host program
        """
        with transactUI(self):
            if self.clientRunner.was_success():
                if self.buildSpec.get('return_to_config', False):
                    self.showSettings()
                else:
                    self.showSuccess()
                    if self.buildSpec.get('show_success_modal', True):
                        wx.CallAfter(modals.showSuccess)
            else:
                if self.clientRunner.wasForcefullyStopped:
                    self.showForceStopped()
                else:
                    self.showError()
                    wx.CallAfter(modals.showFailure)


    def onStopExecution(self):
        """Displays a scary message and then force-quits the executing
        client code if the user accepts"""
        if self.buildSpec['show_stop_warning'] and modals.confirmForceStop():
            self.clientRunner.stop()


    def fetchExternalUpdates(self):
        """
        !Experimental!
        Calls out to the client code requesting seed values to use in the UI
        !Experimental!
        """
        seeds = seeder.fetchDynamicProperties(
            self.buildSpec['target'],
            self.buildSpec['encoding']
        )
        for config in self.configs:
            config.seedUI(seeds)


    def onCancel(self):
        """Close the program after confirming"""
        if modals.confirmExit():
            self.onClose()


    def onClose(self, *args, **kwargs):
        """Cleanup the top level WxFrame and shutdown the process"""
        self.Destroy()
        sys.exit()


    def layoutComponent(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.header, 0, wx.EXPAND)
        sizer.Add(wx_util.horizontal_rule(self), 0, wx.EXPAND)
        sizer.Add(self.navbar, 1, wx.EXPAND)
        sizer.Add(self.console, 1, wx.EXPAND)
        sizer.Add(wx_util.horizontal_rule(self), 0, wx.EXPAND)
        sizer.Add(self.footer, 0, wx.EXPAND)
        # self.SetMinSize((400, 300))
        self.SetSize(self.buildSpec['default_size'])
        self.SetSizer(sizer)
        self.header.Hide()
        self.console.Hide()
        self.footer.Hide()
        self.Layout()
        # self.SetIcon(wx.Icon(self.buildSpec['images']['programIcon'], wx.BITMAP_TYPE_ICO))



    def buildNavigation(self):
        """
        Chooses the appropriate layout navigation component based on user prefs
        """
        if self.buildSpec['navigation'] == constants.TABBED:
            navigation = Tabbar(self, self.buildSpec, self.configs)
        else:
            navigation = Sidebar(self, self.buildSpec, self.configs)
            if self.buildSpec['navigation'] == constants.HIDDEN:
                navigation.Hide()
        return navigation


    def buildConfigPanels(self, parent):
        page_class = TabbedConfigPage if self.buildSpec['tabbed_groups'] else ConfigPage

        return [page_class(parent, widgets)
                for widgets in self.buildSpec['widgets'].values()]


    def showSettings(self):
        self.navbar.Show(True)
        self.console.Show(False)
        self.header.setImage('settings_img')
        self.header.setTitle(_("settings_title"))
        self.header.setSubtitle(self.buildSpec['program_description'])
        self.footer.showButtons('cancel_button', 'start_button')
        self.footer.progress_bar.Show(False)


    def showConsole(self):
        self.navbar.Show(False)
        self.console.Show(True)
        self.header.setImage('running_img')
        self.header.setTitle(_("running_title"))
        self.header.setSubtitle(_('running_msg'))
        self.footer.showButtons('stop_button')
        self.footer.progress_bar.Show(True)
        if not self.buildSpec['progress_regex']:
            self.footer.progress_bar.Pulse()


    def showComplete(self):
        self.navbar.Show(False)
        self.console.Show(True)
        self.footer.showButtons('edit_button', 'restart_button', 'close_button')
        self.footer.progress_bar.Show(False)


    def showSuccess(self):
        self.showComplete()
        self.header.setImage('check_mark')
        self.header.setTitle(_('finished_title'))
        self.header.setSubtitle(_('finished_msg'))
        self.Layout()


    def showError(self):
        self.showComplete()
        self.header.setImage('error_symbol')
        self.header.setTitle(_('finished_title'))
        self.header.setSubtitle(_('finished_error'))


    def showForceStopped(self):
        self.showComplete()
        if self.buildSpec.get('force_stop_is_error', True):
            self.showError()
        else:
            self.showSuccess()
        self.header.setSubtitle(_('finished_forced_quit'))


class DoublePage(wx.Panel):
    def __init__(self, parser_1, parser_2, title_1, title_2, *args, **kwds):
        super(DoublePage, self).__init__(*args, **kwds)
        self.SetSize(610*2, 530)

        spec_1 = build_spec_from_parser(
            parser_1,
            sidebar_title=title_1,
            **kwds)

        spec_2 = build_spec_from_parser(
            parser_2,
            sidebar_title=title_2,
            **kwds)

        self.panel_1 = Page(spec_1, parent=self, id=wx.ID_ANY)
        # self.panel_1 = wx.Button(self, wx.ID_ANY, 'panel1')
        self.panel_2 = Page(spec_2, parent=self, id=wx.ID_ANY)
        # self.panel_2 = wx.Button(self, wx.ID_ANY, 'panel2')

        self.__do_layout()

    def __do_layout(self):
        # sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1.Add(self.panel_1, 1, wx.EXPAND, 5)
        sizer_1.Add(self.panel_2, 1, wx.EXPAND, 5)
        # self.SetSizeHints(self)
        self.SetSizer(sizer_1)
        # self.panel_left.navbar.Hide()
        self.Layout()


class TriplePage(wx.Panel):
    def __init__(self, parser_1, parser_2, parser_3,
                 title_1, title_2, title_3, *args, **kwds):
        super(TriplePage, self).__init__(*args, **kwds)
        self.SetSize(610*2, 530)

        print(parser_1)
        print(parser_2)
        print(parser_3)

        spec_1 = build_spec_from_parser(
            parser_1,
            sidebar_title=title_1,
            **kwds)

        spec_2 = build_spec_from_parser(
            parser_2,
            sidebar_title=title_2,
            **kwds)

        spec_3 = build_spec_from_parser(
            parser_3,
            sidebar_title=title_3,
            **kwds)

        self.panel_1 = Page(spec_1, parent=self, id=wx.ID_ANY)
        # self.panel_1 = wx.Button(self, wx.ID_ANY, 'panel1')
        self.panel_2 = Page(spec_2, parent=self, id=wx.ID_ANY)
        # self.panel_2 = wx.Button(self, wx.ID_ANY, 'panel2')
        self.panel_3 = Page(spec_3, parent=self, id=wx.ID_ANY)

        self.__do_layout()

    def __do_layout(self):
        # sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1.Add(self.panel_1, 1, wx.EXPAND, 5)
        sizer_1.Add(self.panel_2, 1, wx.EXPAND, 5)
        sizer_1.Add(self.panel_3, 1, wx.EXPAND, 5)
        # self.SetSizeHints(self)
        self.SetSizer(sizer_1)
        # self.panel_left.navbar.Hide()
        self.Layout()
