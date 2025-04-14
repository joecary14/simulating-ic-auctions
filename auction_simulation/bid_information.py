class BidInformation:
    def __init__(
        self,
        id : int,
        bid_price : float,
        bid_capacity : float
    ):
        self.id = id
        self.bid_price = bid_price
        self.bid_capacity = bid_capacity