from __future__ import annotations

from networks import Network, Layer

from random import randint
import numpy as np
from math import sin, cos, sqrt, fabs

class GAPopulation:
    def __init__(self, population: int, genes: int):
        self.population_count = population
        self.genes_per_specimen = genes
        
        self.survivor_count = 10
        self.keep_elite = False

        self.genepool = np.random.uniform(-1, 1, (self.population_count, self.genes_per_specimen))

        self.best_gene = self.genepool[0]
        self.best_val = 0
    
    def genetic_selection(self, fitness: np.ndarray) -> None:
        best_index = np.argmax(fitness)
        best_gene_in_generation = self.genepool[best_index]
        if fitness[best_index] > self.best_val:
            self.best_val = fitness[best_index]
            self.best_gene = best_gene_in_generation
        
        argpart = np.argpartition(fitness, -self.survivor_count)
        extermination_indices = argpart[:-self.survivor_count]
        survivor_indices = argpart[-self.survivor_count:]
        for index in extermination_indices:
            parent_a = survivor_indices[randint(0, self.survivor_count-1)]
            parent_b = survivor_indices[randint(0, self.survivor_count-1)]
            crossover_point = randint(0, self.genes_per_specimen - 1)
            # Kinda equivalent to combining the binary strings
            self.genepool[index,:crossover_point] = self.genepool[parent_a,:crossover_point]
            self.genepool[index,crossover_point:] = self.genepool[parent_b,crossover_point:]
            self.genepool[index,crossover_point] += np.random.uniform(-1, 1, 1)[0]
        
        mutation_filter = np.random.uniform(0, 1, self.genepool.shape) < 0.01
        self.genepool[mutation_filter] += np.random.normal(0, 1, self.genepool[mutation_filter].shape)
        self.genepool = np.mod(self.genepool + 1, 2) - 1

        if self.keep_elite:
            self.genepool[best_index] = best_gene_in_generation


class GANeuralNets(GAPopulation):
    def __init__(self, population: int, p_architecture: list[Layer], p_scale: float=1):
        self.networks = []
        self.scale = p_scale
        self.architecture = p_architecture
        for i in range(population):
            self.networks.append(Network(p_architecture, (-p_scale, p_scale), (-p_scale, p_scale)))
        
        super().__init__(population, len(self.networks[0].weights))
    
    def genetic_selection(self, fitness: np.ndarray) -> None:
        super().genetic_selection(fitness)
        for i in range(self.population_count):
            self.networks[i].weights = self.genepool[i] * self.scale


    def process_inputs(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.zeros((len(inputs), self.networks[0].architecture[-1].size))
        for i, inp in enumerate(inputs):
            outputs[i] = self.networks[i].eval(inp).flatten()
        return outputs
    
    def get_best_network(self) -> Network:
        res = Network(self.architecture, (-self.scale, self.scale), (-self.scale, self.scale))
        res.weights = self.best_gene * self.scale
        return res



def test_ga():
    pop = GAPopulation(40, 2)

    for i in range(20):
        x1 = pop.genepool[:,0] * 512
        x2 = pop.genepool[:,1] * 512
        
        # eggholder
        a = np.sqrt(np.fabs(x2 + x1/2 + 47))
        b = np.sqrt(np.fabs(x1 - (x2 + 47)))
        fitness = (x2 + 47)*np.sin(a) - x1*np.sin(b)

        pop.genetic_selection(fitness)

    print(pop.best_val, pop.best_gene)
    
if __name__ == "__main__":
    test_ga()
