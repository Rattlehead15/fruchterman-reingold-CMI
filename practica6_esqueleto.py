#! /usr/bin/python

# 6ta Practica Laboratorio 
# Complementos Matematicos I
# Ejemplo parseo argumentos

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
import numpy as np
import json


class LayoutGraph:

    def __init__(self, grafo, iters, refresh, c1, c2, grav, temp, cooling, verbose, save_to):
        """
        Parámetros:
        grafo: grafo en formato lista
        iters: cantidad de iteraciones a realizar
        refresh: cada cuántas iteraciones graficar. Si su valor es cero, entonces debe graficarse solo al final.
        c1: constante de repulsión
        c2: constante de atracción
        grav: constante gravitatoria
        temp: temperatura inicial
        cooling: constante de enfriamiento
        verbose: si está encendido, activa los comentarios
        save_to: guarda el plot en el archivo especificado
        """

        # Guardo el grafo
        self.grafo = grafo

        # Inicializo estado
        self.posiciones = {v : np.random.rand(2) * 100 for v in grafo[0]}
        self.fuerzas = {v : np.zeros(2) for v in grafo[0]}

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        self.refresh = refresh
        self.t = temp
        self.ct = cooling
        self.g = grav

        ratio_area_vertices_sqrt = np.sqrt(10000 / len(grafo[0]))
        self.k1 = c1 * ratio_area_vertices_sqrt
        self.k2 = c2 * ratio_area_vertices_sqrt
        self.save_to = save_to

    def layout(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout
        """
        for i in range(self.iters):
            if self.verbose:
                print(f"---Iteracion {i}---")
            self.step()
            if self.refresh != 0 and (i % self.refresh) == 0:
                if self.verbose:
                    print(f"Mostrando estado luego de iteracion {i}")
                self.prepare_plot()
                plt.show()
        if self.verbose:
            print("Mostrando estado final")
        self.prepare_plot()
        if self.save_to:
            plt.savefig(self.save_to)
        else:
            plt.show()

    def layout_anim(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout, mostrando los pasos en una animacion
        """
        (fig, scat, lc) = self.prepare_plot()
        ani = animation.FuncAnimation(fig, self.plot_step, frames=self.iters, fargs=[scat, lc], interval=100, repeat=False)
        if self.save_to:
            ani.save(self.save_to)
        else:
            plt.show()
    
    def prepare_plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        scat = ax.scatter(*zip(*self.posiciones.values()))
        edges_plot = [(self.posiciones[v1], self.posiciones[v2]) for (v1, v2) in self.grafo[1]]
        lc = collections.LineCollection(edges_plot)
        ax.add_collection(lc)
        for (v, pos) in self.posiciones.items():
            plt.annotate(v, pos, textcoords='offset points', xytext=(0, 10), ha='center')
        return (fig, scat, lc)

    def plot_step(self, it, scat, lc):
        if self.verbose:
            print(f"---Iteracion {it}---")
        self.step()
        scat.set_offsets(list(self.posiciones.values()))
        edges_plot = [(self.posiciones[v1], self.posiciones[v2]) for (v1, v2) in self.grafo[1]]
        lc.set_segments(edges_plot)
        return scat

    def step(self):
        self.initialize_accumulators()
        self.compute_attraction_forces()
        self.compute_repulsion_forces()
        self.compute_gravity()
        self.update_positions()
        self.t *= self.ct

    def initialize_accumulators(self):
        self.acum = {v : (0, 0) for v in self.grafo[0]}
        if self.verbose:
            print("Acumuladores inicializados")
            print(f"Temperatura actual: {self.t}")

    def compute_attraction_forces(self):
        for v1, v2 in self.grafo[1]:
            edge_vector = self.posiciones[v2] - self.posiciones[v1]
            distance = np.linalg.norm(edge_vector)
            mod_fa_over_distance = distance / self.k2 # Se cancela un distance
            force_vector = mod_fa_over_distance * edge_vector
            if self.verbose:
                print(f"Vector atraccion de {v1} a {v2}: {force_vector}")
            self.acum[v1] += force_vector
            self.acum[v2] -= force_vector

    def compute_repulsion_forces(self):
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2:
                    edge_vector = self.posiciones[v2] - self.posiciones[v1]
                    distance = np.linalg.norm(edge_vector)
                    if distance < 0.05:
                        if self.verbose:
                            print("Distancia cercana a 0, aplicando fuerza de repulsion")
                        self.acum[v1] -= 1
                        self.acum[v2] += 1
                    else:
                        mod_fr_over_distance = (self.k1 / distance) ** 2
                        force_vector = mod_fr_over_distance * edge_vector
                        if self.verbose:
                            print(f"Vector repulsion de {v1} a {v2}: {force_vector}")
                        self.acum[v1] -= force_vector
                        self.acum[v2] += force_vector
    
    def compute_gravity(self):
        for v in self.grafo[0]:
            grav_vector = (50, 50) - self.posiciones[v]
            distance = np.linalg.norm(grav_vector)
            grav_vector *= self.g / distance
            if self.verbose:
                print(f"Gravedad en {v}: {grav_vector}")
            self.acum[v] += grav_vector

    def update_positions(self):
        if self.verbose:
            print("Nuevas posiciones:")
        for v in self.grafo[0]:
            magnitude = np.linalg.norm(self.acum[v])
            if magnitude >= self.t:
                self.acum[v] *= self.t / magnitude
            self.posiciones[v] += self.acum[v]
            np.clip(self.posiciones[v], 0, 100, out=self.posiciones[v])
            if self.verbose:
                print(f"\t{v}: {self.posiciones[v]}")


def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Muestra mas informacion al correr el programa'
    )
    # Mostrar plot animado, opcional, False por defecto
    parser.add_argument(
        '-a', '--animated',
        action='store_true',
        help='Muestra el plot como una animacion en vez de imagenes estaticas'
    )
    # Guardar resultado en un archivo, se debe especificar un nombre
    parser.add_argument(
        '-s', '--save_to',
        help='Guarda el resultado en un archivo'
    )
    # Cantidad de iteraciones, opcional, 50 por defecto
    parser.add_argument(
        '--iters',
        type=int,
        help='Cantidad de iteraciones a efectuar',
        default=50
    )
    # Cantidad de frames hasta proximo plot, opcional, 0 por defecto
    parser.add_argument(
        '--refresh',
        type=int,
        help='Cantidad de frames hasta proximo plot',
        default=0
    )
    # Temperatura inicial
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=100.0
    )
    # Constante de enfriamiento (que tan rapido baja la temperatura)
    parser.add_argument(
        '--cooling',
        type=float,
        help='Constante que modifica la velocidad de enfriamiento',
        default=0.9
    )
    # Archivo del cual leer el grafo
    parser.add_argument(
        'file',
        type=argparse.FileType("r"),
        help='Archivo del cual leer el grafo a dibujar'
    )
    # Constante de repulsion
    parser.add_argument(
        '--repulsion',
        type=float,
        help='Constante que modifica la fuerza de repulsion',
        default=0.1
    )
    # Constante de atraccion
    parser.add_argument(
        '--attraction',
        type=float,
        help='Constante que modifica la fuerza de atraccion (es mas fuerte si es cercana a 0)',
        default=5
    )
    # Constante de gravedad
    parser.add_argument(
        '--gravity',
        type=float,
        help='Constante que modifica la fuerza de gravedad',
        default=3
    )

    args = parser.parse_args()
    grafo = json.load(args.file)

    # Creamos nuestro objeto LayoutGraph
    layout_gr = LayoutGraph(
        grafo,
        iters=args.iters,
        refresh=args.refresh,
        c1=args.repulsion,
        c2=args.attraction,
        verbose=args.verbose,
        grav=args.gravity,
        temp=args.temp,
        cooling=args.cooling,
        save_to=args.save_to
    )

    # Ejecutamos el layout
    if args.animated:
        layout_gr.layout_anim()
    else:
        layout_gr.layout()
    return


if __name__ == '__main__':
    main()
