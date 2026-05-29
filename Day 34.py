class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None


    def insert(self, value):
        self.root = self._insert(self.root, value)

    def _insert(self, node, value):
        if node is None:
            return TreeNode(value)
        elif value < node.value:
            node.left = self._insert(node.left, value)
        else:
            node.right = self._insert(node.right, value)
        return node
    
    def search(self, value):
        return self._search(self.root, value)
    def _search(self, node, value):
        if node is None:
            return False
        elif value == node.value:
            return True
        elif value < node.value:
            return self._search(node.left, value)
        else:
            return self._search(node.right, value)
        
    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result
    def _inorder(self, node, result):
        if node is None:
            return []
        self._inorder(node.left, result)
        result.append(node.value)
        self._inorder(node.right, result)

    def preorder(self):
        result = []
        self._preorder(self.root, result)
        return result
    def _preorder(self, node, result):
        if node is None:
            return []
        result.append(node.value)
        self._preorder(node.left, result)
        self._preorder(node.right, result)
    
    def postorder(self):
        result = []
        self._postorder(self.root, result)
        return result
    def _postorder(self, node, result):
        if node is None:
            return []
        self._postorder(node.left, result)
        self._postorder(node.right, result)
        result.append(node.value)

    def height(self):
        return self._height(self.root)
    def _height(self, node):
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def count(self):
        return self._count(self.root)
    def _count(self, node):
        if node is None:
            return 0
        return 1 + self._count(node.left) + self._count(node.right)
    
    def minimum(self):
        return self._minimum(self.root)
    def _minimum(self, node):
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node.value
    
    def maximum(self):
        return self._maximum(self.root)
    def _maximum(self, node):
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node.value
    
    def __contains__(self, value):
        return self.search(value)
    
    def __len__(self):
        return self.count()
    
    def __str__(self):
        return f"BST | {len(self)} nodes | height: {self.height()}"
    
# Example usage
bst = BST()
for val in [10, 5, 15, 3, 7, 12, 20]:
    bst.insert(val)

print(bst)                  # BST | 7 nodes | height: 3
print(bst.inorder())        # [3, 5, 7, 10, 12, 15, 20]
print(bst.preorder())       # [10, 5, 3, 7, 15, 12, 20]
print(bst.postorder())      # [3, 7, 5, 12, 20, 15, 10]
print(10 in bst)            # True
print(99 in bst)            # False
print(bst.minimum())        # 3
print(bst.maximum())        # 20
print(bst.height())         # 3
print(len(bst))             # 7