import os
from importlib import import_module

from types import ModuleType
#  from Pathlib import Path
import wx


class ModuleTree(wx.TreeCtrl):
    def __init__(self, root_module: ModuleType,
                 *args, **kwds):
        super(ModuleTree, self).__init__(*args, **kwds)
        self.root_module = root_module
        self.root_id = self.AddRoot(text=root_module.__name__)

        self.__do_build()
        self.__do_binds()

    def __do_build(self):
        #  root_id = self.tree.root
        #  root_node = self.tree.get_node(root_id)
        #  root_tag = root_node.tag

        self.SetItemData(self.root_id, self.root_module)
        self.extendTree(parent_item_id=self.root_id)
        self.Expand(self.root_id)

    def __do_binds(self):
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.TreeOnActivated)
        self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.TreeOnExpand)

    def extendTree(self, parent_item_id):
        parent_module = self.GetItemData(parent_item_id)
        print('parent', parent_module.__name__)

        if hasattr(parent_module, '__all__'):
            #  from parent_module import *
            #  child_modules = [module
            #                   for k, module in parent_module.__dict__.items()
            #                   if k in parent_module.__all__]
            child_modules = [
                    import_module('.'.join(
                        [parent_module.__package__, m]))
                    for m in parent_module.__all__]

            for child_module in child_modules:
                child_item_id = self.AppendItem(
                    parent=parent_item_id,
                    text=child_module.__name__.split('.')[-1])
                self.SetItemData(child_item_id, child_module)

                #  self.ExtendTree(item_id)

                if hasattr(child_module, '__all__'):
                    #  from child_module import *
                    #  grand_child_modules = [
                    #          module
                    #          for k, module in child_module.__dict__.items()
                    #          if k in child_module.__all__]
                    grand_child_modules = [
                            import_module('.'.join(
                                [child_module.__package__, m]))
                            for m in child_module.__all__]

                    for grand_child_module in grand_child_modules:
                        grand_child_item_id = self.AppendItem(
                            parent=child_item_id,
                            text=grand_child_module.__name__.split('.')[-1])
                        self.SetItemData(
                                grand_child_item_id, grand_child_module)

    def TreeOnActivated(self, event):
        item_id = event.GetItem()
        module = self.GetItemData(item_id)
        print(module)

    def TreeOnExpand(self, event):
        item_id = event.GetItem()

        child_item_id, cookie = self.GetFirstChild(item_id)
        while child_item_id.IsOk():
            self.DeleteChildren(item=child_item_id)
            self.extendTree(parent_item_id=child_item_id)

            child_item_id, cookie = self.GetNextChild(item_id, cookie)


# class TreeFrame(wx.Frame):
#    def __init__(self, *args, **kwds):
#        super(TreeFrame, self).__init__(*args, **kwds)
#        self.test_tree = Tree(self)

# if __name__ == '__main__':
#    app = wx.App(False)
#    frame = TreeFrame(None)
#    frame.Show()
#    app.MainLoop()
