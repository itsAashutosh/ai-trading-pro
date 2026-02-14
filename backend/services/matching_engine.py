class MatchingEngine:
    """
    Simulates an order matching engine.
    In a real system, this would match Buy/Sell orders between users.
    For now, it acts as a liquidity provider (always fills at market price).
    """
    
    @staticmethod
    def match_order(order_details):
        """
        Match an order against the order book.
        
        Args:
            order_details (dict): Order info
            
        Returns:
            dict: Execution report
        """
        # Simulation: Instant fill at market price
        return {
            'status': 'filled',
            'filled_quantity': order_details['quantity'],
            'filled_price': order_details['price'],
            'timestamp': 'now'
        }
