from flask import Flask
from flask_mysqldb import MySQL
from flask import Blueprint, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'localhost' or ''
app.config['MYSQL_HOST'] = 'localhost'


mysql = MySQL()

@app.route('/')
def Index():
    return render_template('home.html')

# @employees.route('/')
# def Index():
#     cur = mysql.connection.cursor()
#     cur.execute('SELECT * FROM empresa')
#     data = cur.fetchall()
#     cur.close()
#     return render_template('index.html', employees=data)


if __name__ == '__main__':
    app.run(port = 4000, debug = True)
