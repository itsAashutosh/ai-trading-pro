from backend import create_app
from backend.shutdown import register_signal_handlers

app = create_app()
register_signal_handlers(app)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
