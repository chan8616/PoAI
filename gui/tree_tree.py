import wx
from treelib import Tree


class TreeTree(wx.TreeCtrl):
    def __init__(self, tree: Tree, *args, **kwds):
        super(TreeTree, self).__init__(*args, **kwds)
        self.tree = tree
        self.__do_build()
        self.__do_binds()

    def __do_build(self):
        root_id = self.tree.root
        root_node = self.tree.get_node(root_id)
        root_tag = root_node.tag

        self.root_id = self.AddRoot(text=root_tag)
        self.SetItemData(self.root_id, root_node)

        self.ExtendTree(parent_item_id=self.root_id)

    def __do_binds(self):
        self.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.TreeOnActivated)
        self.Bind(wx.EVT_TREE_ITEM_EXPANDING, self.TreeOnExpand)

    def ExtendTree(self, parent_item_id):
        node = self.GetItemData(parent_item_id)
        for node in self.tree.children(node.identifier):
            item_id = self.AppendItem(
                parent=parent_item_id, text=node.tag+str(node))
            self.SetItemData(item_id, node)

            self.ExtendTree(item_id)

    def TreeOnActivated(self, event):
        item_id = event.GetItem()
        node = self.GetItemData(item_id)
        print(node)

    def TreeOnExpand(self, event):
        item_id = event.GetItem()

        child_item_id, cookie = self.GetFirstChild(item_id)
        while child_item_id.IsOk():
            self.DeleteChildren(item=child_item_id)
            self.ExtendTree(parent_item_id=child_item_id)

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
