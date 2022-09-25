from app import app
from employees import employees

app.register_blueprint(employees)

# starting the app
if __name__ == "__main__":
    app.run(port=3019, debug=True)