import pandas as pd
import mysql.connector

db = mysql.connector.connect(user = 'root', database = 'hackmty', password = '')
cursor = db.cursor()

query = "select nombre , email from empresa"
cursor.execute(query)

myallData = cursor.fetchall()

all_id_empresa = []
all_nombre = []
all_longitud = []
all_latitud = []

for  id_empresa, nombre, longitud, latitud in myallData:
    all_id_empresa.append(id_empresa)
    all_nombre.append(nombre)
    all_longitud.append(longitud)
    all_latitud.append(latitud)

#we need to store this data to CSV
dic = {'id_empresa' : all_id_empresa, 'nombre' : all_nombre, 'all_longitud' : all_longitud, 'all_latitud' : all_latitud}
df = pd.DataFrame (dic)
df_csv = df.to_csv('archivoDeDB.csv')

