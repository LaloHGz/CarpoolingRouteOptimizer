import pandas as pd
import mysql.connector

db = mysql.connector.connect(user = 'root', database = 'hackmty', password = '')
cursor = db.cursor()

queryTrabajador = "select id_empresa, nombre, longitud, latitud from trabajador"
cursor.execute(queryTrabajador)
trabajadorData = cursor.fetchall()

all_Vol = []
all_nombre = []
all_longitud = []
all_latitud = []

for  id_empresa, nombre, longitud, latitud in trabajadorData:
    all_Vol.append(1)
    all_nombre.append(nombre)
    all_longitud.append(longitud)
    all_latitud.append(latitud)

dic = {'nombre' : all_nombre, 'Lat' : all_latitud, 'Lon' : all_longitud, "Volumen" : all_Vol}
df = pd.DataFrame (dic)
df_csv = df.to_csv('coords.csv')
# -------------------------- ---------------------------- -----------------------

queryVehiculo = "select id_empresa, modelo, capacidad, rendimiento from vehiculo"
cursor.execute(queryVehiculo)
vehiculoData = cursor.fetchall()

all_id_vehiculo = []
all_modelo = []
all_capacidad = []
all_rendimiento = []

for  id_empresa, modelo, capacidad, rendimiento in vehiculoData:
    all_id_vehiculo.append(id_empresa)
    all_modelo.append(modelo)
    all_capacidad.append(capacidad)
    all_rendimiento.append(rendimiento)

dic = {'Nom_Tipo_Unidad' : all_id_vehiculo, 'Modelo' : all_modelo, 
       'Volumen de caja (cuanto puede cargar)' : all_capacidad, 'rendimiento (kilometro/litro)' : all_rendimiento}
df = pd.DataFrame (dic)
df_csv = df.to_csv('flota.csv')