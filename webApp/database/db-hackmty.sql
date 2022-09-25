-- creating database
CREATE DATABASE hackmty;

-- using the database
use hackmty;

-- creating table
CREATE TABLE empresa (
    id_empresa INT(6) UNSIGNED AUTO_INCREMENT,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL,
    contrasena VARCHAR(100) NOT NULL,
    PRIMARY KEY (id_empresa)
);

CREATE TABLE trabajador (
    id_trabajador INT(6) UNSIGNED AUTO_INCREMENT,
    id_empresa INT(6) UNSIGNED NOT NULL,
    nombre VARCHAR(50) NOT NULL,
    longitud VARCHAR(100) NOT NULL,
    latitud VARCHAR(150) NOT NULL,
    PRIMARY KEY (id_trabajador),
    FOREIGN KEY (id_empresa) REFERENCES empresa(id_empresa) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE vehiculo (
    id_vehiculo INT(6) UNSIGNED AUTO_INCREMENT,
    id_empresa INT(6) UNSIGNED NOT NULL,
    modelo VARCHAR(50) NOT NULL,
    capacidad INT(2) NOT NULL,
    rendimiento INT(3) NOT NULL,
    PRIMARY KEY (id_vehiculo),
    FOREIGN KEY (id_empresa) REFERENCES empresa(id_empresa) ON DELETE CASCADE ON UPDATE CASCADE
);


CREATE TABLE itinerario (
    id_itinerio INT(6) UNSIGNED AUTO_INCREMENT,
    id_empresa INT(6) UNSIGNED NOT NULL,
    id_trabajador INT(6) UNSIGNED NOT NULL,
    hora VARCHAR(50) NOT NULL,
    PRIMARY KEY (id_itinerio),
    FOREIGN KEY (id_empresa) REFERENCES empresa(id_empresa),
    FOREIGN KEY (id_trabajador) REFERENCES trabajador(id_trabajador) ON DELETE CASCADE ON UPDATE CASCADE
);