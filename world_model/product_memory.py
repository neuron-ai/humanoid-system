class ProductMemory:

    def __init__(self):

        self.products = {}

    def add_product(self, name, metadata):

        self.products[name] = metadata

    def get(self, name):

        return self.products.get(name)

    def all(self):

        return self.products