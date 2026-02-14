import requests
from datetime import datetime
from flask import current_app
from backend.extensions import db
from backend.models.trading import Holding

def fetch_current_price(symbol):
    """Fetches the latest real-time stock price from Finnhub."""
    api_key = current_app.config.get('FINNHUB_API_KEY')
    base_url = current_app.config.get('FINNHUB_BASE_URL')
    
    if not api_key or api_key == 'YOUR_FINNHUB_API_KEY_HERE':
        print(f"⚠️ WARNING: Finnhub API key is not set. Cannot fetch live data for {symbol}.")
        return None

    # Finnhub 'quote' endpoint
    url = f"{base_url}/quote?symbol={symbol}&token={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # 'c' is the current price
        if data and data.get('c') is not None and data.get('c') > 0:
            current_price = data['c']
            return current_price
        else:
            print(f"🛑 Error: Invalid or missing data for {symbol} from Finnhub API. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch network errors or HTTP errors
        print(f"❌ Error fetching live data for {symbol}: {e}")
        return None

from backend.tasks.job_guard import monitor_job

@monitor_job("Price Update Job")
def update_all_holdings_prices(app):
    """Scheduled task to update the current price for all unique holdings."""
    # This must run within the Flask application context to interact with the database
    with app.app_context():
        print(f"\n🔄 Running scheduled price update at {datetime.now().strftime('%H:%M:%S')}...")
        
        # Get all unique symbols currently held by any user
        unique_symbols = db.session.query(Holding.symbol).distinct().all()
        symbols_to_update = [s[0] for s in unique_symbols]
        
        if not symbols_to_update:
            print("ℹ️ No holdings found to update.")
            return

        print(f"Updating prices for symbols: {symbols_to_update}")
        
        # Fetch prices for all symbols
        for symbol in symbols_to_update:
            live_price = fetch_current_price(symbol)
            
            if live_price is not None:
                # Update all holdings with this symbol across all users
                updated_count = db.session.query(Holding).filter(
                    Holding.symbol == symbol
                ).update(
                    {'current_price': live_price, 'last_updated': datetime.utcnow()},
                    synchronize_session='fetch'
                )
                if updated_count > 0:
                    print(f"   -> Successfully updated {updated_count} holding(s) for {symbol} to ${live_price:.2f}")
        
        try:
            db.session.commit()
            print("✅ Database commit successful.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Database transaction failed during price update: {e}")
