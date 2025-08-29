from flask import Flask
from auth import auth
from main import main

app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for session & flash

# Register blueprints
app.register_blueprint(auth, url_prefix="/auth")
app.register_blueprint(main, url_prefix="/")

if __name__ == "__main__":
    app.run(debug=True)
