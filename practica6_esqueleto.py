#! /usr/bin/python

# 6ta Practica Laboratorio 
# Complementos Matematicos I
# Ejemplo parseo argumentos

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as collections
import numpy as np


class LayoutGraph:

    def __init__(self, grafo, iters, refresh, c1, c2, verbose=False):
        """
        Parámetros:
        grafo: grafo en formato lista
        iters: cantidad de iteraciones a realizar
        refresh: cada cuántas iteraciones graficar. Si su valor es cero, entonces debe graficarse solo al final.
        c1: constante de repulsión
        c2: constante de atracción
        verbose: si está encendido, activa los comentarios
        """

        # Guardo el grafo
        self.grafo = grafo

        # Inicializo estado
        # Completar
        self.posiciones = {v : np.random.rand(2) * 100 for v in grafo[0]}
        self.fuerzas = {v : np.zeros(2) for v in grafo[0]}

        # Guardo opciones
        self.iters = iters
        self.verbose = verbose
        # TODO: faltan opciones
        # agregar parametro para g (constante de gravedad)
        self.refresh = refresh
        self.c1 = c1
        self.c2 = c2
        self.g = 2

        ratio_area_vertices_sqrt = np.sqrt(10000 / len(grafo[0]))
        self.k1 = c1 * ratio_area_vertices_sqrt
        self.k2 = c2 * ratio_area_vertices_sqrt

    def layout(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout
        """
        for _ in range(self.iters):
            self.step()
        
        self.prepare_plot()
        plt.show()

    def layout_anim(self):
        """
        Aplica el algoritmo de Fruchtermann-Reingold para obtener (y mostrar)
        un layout, mostrando los pasos en una animacion
        """
        (fig, scat, lc) = self.prepare_plot()
        anim = animation.FuncAnimation(fig, self.plot_step, frames=self.iters, fargs=[scat, lc], interval=100)
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

    def plot_step(self, iter, scat, lc):
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

    def initialize_accumulators(self):
        self.acum = {v : (0, 0) for v in self.grafo[0]}

    def compute_attraction_forces(self):
        for v1, v2 in self.grafo[1]:
            edge_vector = self.posiciones[v2] - self.posiciones[v1]
            distance = np.linalg.norm(edge_vector)
            mod_fa_over_distance = distance / self.k2 # Se cancela un distance
            force_vector = mod_fa_over_distance * edge_vector
            self.acum[v1] += force_vector
            self.acum[v2] -= force_vector

    def compute_repulsion_forces(self):
        for v1 in self.grafo[0]:
            for v2 in self.grafo[0]:
                if v1 != v2:
                    edge_vector = self.posiciones[v2] - self.posiciones[v1]
                    distance = np.linalg.norm(edge_vector)
                    mod_fr_over_distance = (self.k1 / distance) ** 2 # Se cancela un distance
                    force_vector = mod_fr_over_distance * edge_vector
                    self.acum[v1] -= force_vector
                    self.acum[v2] += force_vector
    
    def compute_gravity(self):
        for v in self.grafo[0]:
            grav_vector = (50, 50) - self.posiciones[v]
            distance = np.linalg.norm(grav_vector)
            self.acum[v] += (self.g / distance) * grav_vector

    def update_positions(self):
        for v in self.grafo[0]:
            self.posiciones[v] += self.acum[v]
        
    def f_attraction(self, d):
        """
        Calcula la fuerza de atraccion entre dos vertices
        """
        return d * d / self.k2
    
    def f_repulsion(self, v1, v2):
        """
        Calcula la fuerza de repulsion entre dos vertices
        """
        return self.posiciones[v1] - self.posiciones[v2]


def main():
    # Definimos los argumentos de linea de comando que aceptamos
    parser = argparse.ArgumentParser()

    # Verbosidad, opcional, False por defecto
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Muestra mas informacion al correr el programa'
    )
    # Cantidad de iteraciones, opcional, 50 por defecto
    parser.add_argument(
        '--iters',
        type=int,
        help='Cantidad de iteraciones a efectuar',
        default=50
    )
    # Temperatura inicial
    parser.add_argument(
        '--temp',
        type=float,
        help='Temperatura inicial',
        default=100.0
    )
    # Archivo del cual leer el grafo
    # parser.add_argument(
    #     'file_name',
    #     help='Archivo del cual leer el grafo a dibujar'
    # )
    # TODO: Descomentar y obligar a pasar archivo

    args = parser.parse_args()

    # Descomentar abajo para ver funcionamiento de argparse
    # print args.verbose
    # print args.iters    
    # print args.file_name
    # print args.temp
    # return

    # TODO: Borrar antes de la entrega
    grafo1 = ([1, 2, 3, 4, 5, 6, 7],
              [(1, 2), (2, 3), (3, 1), (5, 6), (6, 7), (7, 5)])

    # Creamos nuestro objeto LayoutGraph
    layout_gr = LayoutGraph(
        grafo1,  # TODO: Cambiar para usar grafo leido de archivo
        iters=args.iters,
        refresh=1,
        c1=0.1,
        c2=5.0,
        verbose=args.verbose
    )

    # Ejecutamos el layout
    layout_gr.layout_anim()
    return


if __name__ == '__main__':
    main()
