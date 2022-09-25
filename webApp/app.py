from flask import Flask
from flask_mysql import MySQL

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'localhost'
app.config['MYSQL_HOST'] = 'localhost'


mysql = MySQL()

@app.route('/')
def Index():
    return 'hello world'

@app.route('/add_worker')
def addWorker():
    return 'add worker'

@app.route('/update_worker')
def updateWorker():
    return 'update worker'

@app.route('/delete_worker')
def deleteWorker():
    return 'delete worker'

@app.route('/add_company')
def addCompany():
    return 'add company'

@app.route('/addVehicle')
def addVehicle():
    return 'add vehicle'

@app.route('/update_vehicle')
def updateVehicle():
    return 'update vehicle'

@app.route('/delete_vehicle')
def deleteVehicle():
    return 'delete vehicle'


if __name__ == '__main__':
    app.run(port = 4000, debug = True)
