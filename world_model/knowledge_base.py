from datetime import datetime


class KnowledgeBase:

    def __init__(self):

        self.rules = {}

    def is_expired(self, product):

        expiry = product.get("expiry")

        if expiry is None:
            return False

        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")

        return expiry_date < datetime.now()

    def fifo_select(self, products):

        return sorted(products, key=lambda p: p["expiry"])[0]