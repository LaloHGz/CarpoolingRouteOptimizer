import openrouteservice
import folium
from folium.plugins import BeautifyIcon
import pandas as pd
import openrouteservice as ors
import pandas                  as pd
import numpy                   as np
import matplotlib.pyplot       as plt
from datetime import date
#import seaborn                 as sns
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation
from IPython.display import display
import pulp as p
import warnings
warnings.filterwarnings("ignore")

def routesGenerator(df, model, n_clusters, mode):

    nk = n_clusters
    df_K = pd.DataFrame(df['Lat'])
    df_K['Lon'] = df['Lon']
    #df_K

    if (model == 1):
        kmeans = KMeans(n_clusters= nk).fit(df_K)
        df_K['Grupo'] = kmeans.predict(df_K)
    elif (model == 2):
        AgC = AgglomerativeClustering(n_clusters= nk).fit(df_K)
        df_K['Grupo'] = AgC.fit_predict(df_K)
    elif (model == 3):
        sc = SpectralClustering(n_clusters= nk).fit(df_K)
        df_K['Grupo'] = sc.fit_predict(df_K)
    elif (model == 4):
        GM = GaussianMixture(n_components= nk).fit(df_K)
        df_K['Grupo'] = GM.predict(df_K)
    #df_K['Needed_Amount'] = df['cantidad']
    #df_K['volumen'] = df['volxpostal']

    z = folium.Map(location=[25.65240416152182, -100.29108458215048], tiles='cartodbpositron', zoom_start=13)
    colors = ['green', 'red', 'blue', 'yellow', 'purple', 'brown', 'grey', 'pink']
    keyAPI = '5b3ce3597851110001cf624875f5f3899e79410bac3b8d81a1bb6096'
    times = []
    distanceS = []


    #ors_client = 
    results = []
    #if (1==1):
    #    i=0
    for i in range(nk):
        df_99=df_K[df_K['Grupo']==i]
        del df_99['Grupo']
        df_99['ID'] = list(range(len(df_99)))
        df_99["Open_From"] = (len(df_99)*"08:00 ").split()
        df_99["Open_To"] = (len(df_99)*"20:00 ").split()
        df_99['Needed_Amount'] = (len(df_99)*"1 ").split()
        df_99.to_csv('df_99.csv')

        

        # Next load the delivery locations from CSV file at ../resources/data/idai_health_sites.csv
        # ID, Lat, Lon, Open_From, Open_To, Needed_Amount
        deliveries_data = pd.read_csv(
            'df_99.csv',
            index_col="ID",
            parse_dates=["Open_From", "Open_To"]
        )

        for location in deliveries_data.itertuples():
            tooltip = folium.map.Tooltip("<h4><b>ID {}</b></p><p>Supplies needed: <b>{}</b></p>".format(
                location.Index, location.Needed_Amount
            ))

            folium.Marker(
                location=[location.Lat, location.Lon],
                tooltip=tooltip,
                icon=BeautifyIcon(
                    icon_shape='marker',
                    number=int(location.Index),
                    spin=True,
                    text_color=colors[i],
                    background_color="#FFF",
                    inner_icon_style="font-size:12px;padding-top:-5px;"
                )
            ).add_to(z)

            
        depot = [25.65240416152182, -100.29108458215048]

        folium.Marker(
            location=depot,
            icon=folium.Icon(color="green", icon="bus", prefix='fa'),
            setZIndexOffset=1000
        ).add_to(z)

        vehicles = list()
        for idx in range(3):
            vehicles.append(
                ors.optimization.Vehicle(
                    id=i,
                    start=list(reversed(depot)),
                    end=list(reversed(depot)),
                    capacity=[300000] 
                )
            )

        deliveries = list()
        for delivery in deliveries_data.itertuples():
            #if (delivery.volumen /delivery.Needed_Amount < 0.51):
                #print(delivery.volumen /delivery.Needed_Amount)
            #    serviceO=300*delivery.Needed_Amount
            #else:
                #print(delivery.volumen /delivery.Needed_Amount)
            #    serviceO=600*delivery.Needed_Amount
                
            deliveries.append(
                ors.optimization.Job(
                    id=delivery.Index,
                    location=[delivery.Lon, delivery.Lat],
                    service = 60,
                    amount=[delivery.Needed_Amount],
                )
            )

        # Get an API key from https://openrouteservice.org/dev/#/signup
        result = ors.Client(key= keyAPI).optimization(
            jobs=deliveries,
            vehicles=vehicles,
            geometry=True
        )
        
        results.append(result)

        # Add the output to the map
        for color, route in zip([colors[i],colors[i],colors[i]], result['routes']):
            decoded = ors.convert.decode_polyline(route['geometry'])  # Route geometry is encoded
            gj = folium.GeoJson(
                name='Vehicle {}'.format(route['vehicle']),
                data={"type": "FeatureCollection", "features": [{"type": "Feature",
                                                                "geometry": decoded,
                                                                "properties": {"color": color}
                                                                }]},
                style_function=lambda x: {"color": x['properties']['color']}
            )
            gj.add_child(folium.Tooltip(
                """<h4>Vehicle {vehicle}</h4>
                <b>Distance</b> {distance} m <br>
                <b>Duration</b> {duration} secs
                """.format(**route)
            ))
            gj.add_to(z)

        
        #time
        last_step=result['routes'][0]['steps'][-1]
        timeR = last_step['arrival']+last_step.get('service',0)
        #print(timeR/3600)
        times.append(timeR/3600)
        #print(pd.to_datetime(timeR, unit='s'))

        #distance
        last_step=result['routes'][0]['steps'][-1]
        distanceR = last_step['distance']
        #print(timeR/3600)
        distanceS.append(distanceR/1000)
        

    folium.LayerControl().add_to(z)
    totaltime = sum(times)
    totaldistance = sum(distanceS)
    maxtime = max(times)


    
    if (mode == 1):
        return(maxtime, totaldistance)
    elif(mode == 3):
        return (times,totaldistance)
    elif(mode == 2):
        return z
    elif (mode == 4):
        return (times,distanceS)
    elif (mode == 5):
        return results

def routesData(df, nMaxRutas):
    maxRutas = nMaxRutas
    dfporK = pd.DataFrame()
    dfporK['K'] = list(range(1,maxRutas))
    models = ['Kmeans_', 'AgC_', 'Spectre_', 'Gaussian_']

    #Models
    for j in range(4):
        timedf = []
        distancedf = []
        for i in range(1,maxRutas):
            timedf.append(routesGenerator(df,j+1,i,1)[0])
            distancedf.append(routesGenerator(df, j+1,i,1)[1])
        dfporK[models[j]+'Tiempo Max'] = timedf
        dfporK[models[j]+'Distancia total'] = distancedf
    
    return dfporK

def routesCosto(salario, costokilometro, dfroutesdata):
    salarioPorHora = salario
    costokilometro = costokilometro
    dfCopy = dfroutesdata.copy()
    dfCostos = pd.DataFrame()
    dfCostos['K'] = [1,2]

    for j in range(4):
        costo = dfCopy[dfCopy.columns[2*j+1]]*177.525*dfCopy[dfCopy.columns[0]]+dfCopy[dfCopy.columns[2*j+2]]*costokilometro
        dfCostos['Costo modelo ' + str(j+1)] = costo

    dfCostos.columns = ['K', 'Kmeans', 'Agglomerative cluster', 'Spectre cluster', 'Gaussian Mixture']    

    return dfCostos

def eliminateLongRoutes(df):
    dfrutasdata = df
    drutas, ncol = dfrutasdata.shape

    for i in range(ncol):
        indices=[]
        if i%2==1:
            for j in range(len(dfrutasdata[dfrutasdata.columns[i]])):
                if dfrutasdata[dfrutasdata.columns[i]][j]>2:
                    dfrutasdata[dfrutasdata.columns[i]][j] = 999999999999
                    dfrutasdata[dfrutasdata.columns[i+1]][j] = 999999999999

    return dfrutasdata

def seleccionaralgoritmo(df):
    arrporK = np.array(df[df.columns[1:]])
    best = [np.where(arrporK == np.min(arrporK))[0][0] + 1, np.where(arrporK == np.min(arrporK))[1][0] + 1]
    #best[1] = dfporK.columns[best[1]]
    return best    

def itinerariosgenerator(df, algoritmo, grupos):
    result = routesGenerator(df, algoritmo, grupos, 5)
    itinerarios = []
    for i in range(grupos):
        stations = list()
        for route in result[i]['routes']:
            vehicle = list()
            for step in route["steps"]:
                vehicle.append(
                    [
                        step.get("job", "Depot"),  # Station ID
                        step["arrival"],  # Arrival time
                        step["arrival"] + step.get("service", 0),  # Departure time
                        step['location']
                    ]
                )
            stations.append(vehicle)

        df_stations_0 = pd.DataFrame(stations[0], columns=["Station ID", "Arrival", "Departure", 'location'])
        df_stations_0['Arrival'] = pd.to_datetime(df_stations_0['Arrival'], unit='s')
        df_stations_0['Departure'] = pd.to_datetime(df_stations_0['Departure'], unit='s')
        itinerarios.append(df_stations_0)
    return itinerarios

def MixedIntegerProgramming(df, flota, sal, grupos, algoritmo, itinerarios):
    
    #Generando variables para LP
    ruteo = routesGenerator(df, algoritmo, grupos, 4)
    n = len(flota)                                                                  #Camiones
    m = grupos                                                                      #Rutas
    s = sal                                                                         #Salario por hora
    C = [23.09/i for i in list(flota['rendimiento (kilometro/litro)'])]             #Lista con los rendimientos de gasolina en $/km
    T = ruteo[0]                                                                    #Lista de Tiempos de cada ruta                                                  #Lista recortada
    D = ruteo[1]                                                                    #Lista de distancia de cada ruta
    V = list(flota['Volumen de caja (cuanto puede cargar)'])                        #Lista de volumenes de cada camion
    K = [len(itinerarios[i]) for i in range(grupos)]                                #Lista de demandas de volumen de cada ruta

    #Variables para PULP
    camiones = list(range(0,n))
    rutas = list(range(0,m))
    
    #Crea el problema LP
    prob = p.LpProblem("RuteoOptimo", p.LpMinimize)     

    #Crea la variable X, tiene maximo de 5 por ahora
    X=p.LpVariable.dicts("X",[(i,j) for i in camiones for j in rutas],0, 5, p.LpInteger)    
    
    #Funcion Objetivo
    prob+= p.lpSum(X[(i,j)] * (D[j]*C[i] + T[j]*s) for i in camiones for j in rutas) 
     
    #Restricciones
    # for i in camiones:
    #     prob+= p.lpSum(X[(i,j)]*(T[j]) for j in rutas) <= 9

    #Restriccion de cumplimiento de demanda 
    for j in rutas:
        prob+=p.lpSum(X[(i,j)]*V[i] for i in camiones) >= K[j]
        
    #Resolviendo
    prob.solve()
    cost = p.value(prob.objective)
    
    ruteoOptimo=pd.DataFrame({'Camiones':[],'Rutas':[]})
    for i in camiones:
        for j in rutas:
            if(X[(i,j)].varValue>0):
                for k in range(0,int(X[(i,j)].varValue)):
                    ruteoOptimo.loc[len(ruteoOptimo)]=[i,j]
                    
    ruteoOptimo['Volumen'] = [V[int(h)] for h in ruteoOptimo['Camiones']]

    return cost, ruteoOptimo

def generate_results(df, itinerarios, grupos):
    
    for i in range(grupos):
        itinerarios[i]['Ruta'] = i

    itinerariofinal = pd.concat(itinerarios, axis=0)
    
    df['location'] = [[df['Lon'][i], df['Lat'][i]] for i in range(len(df))]
    
    itinerariofinal['location'] = [str(i) for i in itinerariofinal['location']]
    df['location'] = [str(i) for i in df['location']]
    
    itinerariofinalbonito = itinerariofinal.merge(df, on='location')
    
    itinerariofinalbonito.drop(columns=['Station ID', 'Departure', 'location', 'Lat', 'Lon'], inplace=True)
    
    itinerariofinalbonito['Arrival'] = pd.to_datetime(itinerariofinalbonito['Arrival'])
    
    arrival = []
    for i in range(len(itinerariofinalbonito)):
        hora = str(itinerariofinalbonito['Arrival'][i].hour + 5)
        minuto = str(itinerariofinalbonito['Arrival'][i].minute)
        if(len(minuto) == 1):
            minuto = '0' + minuto
        arrival.append(hora + ':' + minuto)
    
    itinerariofinalbonito['Arrival'] = arrival
    
    return itinerariofinalbonito
    

def main():

    #Salario de choferes $/hora
    sal = 100

    #Costo por kilometro (coche de prueba)
    costoKilometro = 22.97/9.57

    #Lee el archivo
    import mysql.connector as connection
    try:
        mydb = connection.connect(host="localhost", database = 'hackmty', user="root", passwd="root", use_pure=True)
        query = "Select * from trabajador;"
        df = pd.read_sql(query,mydb)
        mydb.close() #close the connection
    except Exception as e:
        mydb.close()
        print(str(e))
        
    # df=pd.DataFrame(pd.read_csv('coords.csv'))

    #crea la información de las rutas 
    dfrutasdata= routesData(df, 3)

    #Elimina tiempos mayores a 2 horas
    dfroutesdata = eliminateLongRoutes(dfrutasdata)

    #Obtiene el costo de las rutas 
    dfporK = routesCosto(sal, costoKilometro, dfrutasdata)

    #Selecciona el mejor algoritmo de machine learning
    [grupos, algoritmo] = seleccionaralgoritmo(dfporK)

    #Generando el mapa con las rutas
    routesGenerator(df, algoritmo, grupos, 2).save('map1.html')

    #Generando los itinerarios de orden de rutas
    itinerarios = itinerariosgenerator(df, algoritmo, grupos)
    
    #Leyendo el archivo de flota
    flota = pd.read_csv('Flota.csv')
    
    #Generando la distribucion de camiones
    costo, ruteoOptimo = MixedIntegerProgramming(df, flota, sal, grupos, algoritmo, itinerarios)
    
    #Obteniendo los resultados finales
    results = generate_results(df, itinerarios, grupos)
    
    print(ruteoOptimo)
    print(results)

    
if __name__ == "__main__":
    main()