class TreeNode:
    __slots__ = ['feature', 'threshold', 'left', 'right', 'leaf_value']
    
    def __init__(self, feature=None,threshold=None,left=None,right=None,leaf_value=None):
        self.feature= feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value= leaf_value
     
    def get_leaf(self):
        return self.leaf_value

    def is_leaf(self):
        return self.leaf_value is not None

    def set_children(self,left=None,right=None):
        self.left=left
        self.right=right

    def set_leaf(self,value):
        self.leaf_value=value

    def get_threshold(self):
        return self.threshold

    def get_feature(self):
        return self.feature