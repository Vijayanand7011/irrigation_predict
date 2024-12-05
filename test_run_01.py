# Import the function from interfaces.py
from interfaces import run_flask_app

# Call the function to initialize the Flask app
app = run_flask_app()

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
