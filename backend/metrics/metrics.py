from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Metrics Definitions
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint']
)
TRADE_COUNT = Counter(
    'trades_total', 'Total executed trades', ['symbol', 'action']
)
TRAINING_JOBS = Counter(
    'training_jobs_total', 'Total RL training jobs', ['status']
)
ACTIVE_JOBS = Gauge(
    'active_background_jobs', 'Number of currently running background jobs'
)

def track_request(method, endpoint, status):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()

def track_latency(method, endpoint, duration):
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

def track_trade(symbol, action):
    TRADE_COUNT.labels(symbol=symbol, action=action).inc()

def get_metrics_data():
    return generate_latest(), CONTENT_TYPE_LATEST
