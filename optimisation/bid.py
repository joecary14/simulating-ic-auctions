from dataclasses import dataclass

@dataclass
class Bid:
    bidder_id: int
    price: float
    quantity: float