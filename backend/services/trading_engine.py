from backend.models.user import User
from backend.models.trading import Trade, Holding
from backend.extensions import db

class TradingEngine:
    @staticmethod
    def execute_trade(user_id, trade_data, dry_run=False):
        """
        Execute a trade or simulate it (dry_run).
        
        Args:
            user_id (int): ID of the user
            trade_data (dict): Trade details (symbol, action, quantity, price)
            dry_run (bool): If True, calculate and return results without DB commit or object modification.
            
        Returns:
            dict: Result containing 'status', 'trade', 'new_balance', or 'error'
        """
        # 1. Fetch User
        user = User.query.get(user_id)
        if not user:
            return {'status': 'error', 'error': 'User not found'}
            
        # 2. Validate Data
        required = ['symbol', 'action', 'quantity', 'price']
        if not all(k in trade_data for k in required):
            return {'status': 'error', 'error': 'Missing required fields'}
            
        # 3. Risk Checks (NEW)
        from backend.services.risk_engine import RiskEngine
        passed, reason = RiskEngine.check_pre_trade_risk(user, trade_data)
        if not passed:
            return {'status': 'error', 'error': f"Risk Rejection: {reason}"}
            
        symbol = trade_data['symbol']
        action = trade_data['action'].upper()
        quantity = int(trade_data['quantity'])
        price = float(trade_data['price'])
        
        expected_total_value = quantity * price
        commission = expected_total_value * 0.001
        
        # 3. Logic per Action
        new_balance = user.cash_balance
        holding_update = None
        
        if action == 'BUY':
            total_cost = expected_total_value + commission
            if user.cash_balance < total_cost:
                return {'status': 'error', 'error': 'Insufficient funds'}
            
            new_balance -= total_cost
            
            # Predict Holding State
            holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first()
            if holding:
                total_shares = holding.shares + quantity
                total_cost_basis = (holding.shares * holding.avg_cost) + (quantity * price)
                new_avg_cost = total_cost_basis / total_shares
                holding_update = {
                    'action': 'update',
                    'shares': total_shares,
                    'avg_cost': new_avg_cost
                }
            else:
                holding_update = {
                    'action': 'create',
                    'shares': quantity,
                    'avg_cost': price
                }
                
        elif action == 'SELL':
            holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first()
            if not holding or holding.shares < quantity:
                return {'status': 'error', 'error': 'Insufficient shares'}
                
            new_balance += (expected_total_value - commission)
            remaining_shares = holding.shares - quantity
            
            if remaining_shares == 0:
                holding_update = {'action': 'delete'}
            else:
                holding_update = {
                    'action': 'update',
                    'shares': remaining_shares
                }
        
        else:
             return {'status': 'error', 'error': 'Invalid action'}

        # 4. Construct Trade Record (Simulation)
        trade_record = {
            'user_id': user_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'total_value': expected_total_value,
            'commission': commission
        }

        # 5. Return / Persist
        if dry_run:
            return {
                'status': 'success',
                'mode': 'shadow',
                'new_balance': new_balance,
                'trade': trade_record,
                'holding_update': holding_update
            }
        
        # --- PERMANENT EXECUTION ---
        user.cash_balance = new_balance
        
        # Create Trade Object
        new_trade = Trade(
            user_id=user_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            order_type=trade_data.get('order_type', 'market'),
            time_in_force=trade_data.get('time_in_force', 'day'),
            total_value=expected_total_value,
            commission=commission,
            notes=trade_data.get('notes')
        )
        db.session.add(new_trade)
        
        # Apply Holding Changes
        if action == 'BUY':
            holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first()
            if holding:
                holding.shares = holding_update['shares']
                holding.avg_cost = holding_update['avg_cost']
                holding.current_price = price
            else:
                new_holding = Holding(
                    user_id=user_id,
                    symbol=symbol,
                    company_name=trade_data.get('company_name'),
                    shares=quantity,
                    avg_cost=price,
                    current_price=price
                )
                db.session.add(new_holding)
        
        elif action == 'SELL':
            holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first()
            if holding_update['action'] == 'delete':
                db.session.delete(holding)
            else:
                holding.shares = holding_update['shares']
        
        try:
            db.session.commit()
            
            # AUDIT LOG
            from backend.audit.audit_logger import log_audit_event
            log_audit_event('TRADE_EXECUTED', user_id=user_id, 
                           symbol=symbol, action=action, 
                           quantity=quantity, price=price, 
                           total_value=expected_total_value)
            
            return {
                'status': 'success',
                'message': 'Trade executed successfully',
                'trade': new_trade.to_dict(),
                'new_balance': user.cash_balance
            }
        except Exception as e:
            db.session.rollback()
            return {'status': 'error', 'error': str(e)}
