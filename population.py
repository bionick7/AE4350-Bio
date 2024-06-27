from __future__ import annotations

from networks import Network, Layer

from random import randint, random
import numpy as np
from math import sin, cos, sqrt, fabs

class GAPopulation:
    def __init__(self, population: int, genes: int, **kwargs):
        self.population_count = population
        self.genes_per_specimen = genes
        
        self.mutation_rate = kwargs.get("mutation_rate", 0.001)
        self.mutation_stdev = kwargs.get("mutation_stdev", 1)
        self.survivor_count = kwargs.get("survivor_count", 10)
        self.elitism = kwargs.get("elitism", True)

        self.genepool = np.random.uniform(-1, 1, (self.population_count, self.genes_per_specimen))

        self.best_gene = self.genepool[0]
        self.best_val = 0
    
    def genetic_selection(self, fitness: np.ndarray) -> None:
        best_index = np.argmax(fitness)
        best_gene_in_generation = self.genepool[best_index]
        if fitness[best_index] > self.best_val:
            self.best_val = fitness[best_index]
            self.best_gene = best_gene_in_generation
        

        cumulative_fitness = np.cumsum(fitness - min(fitness)) / np.sum(fitness - min(fitness))
        
        argpart = np.argpartition(fitness, -self.survivor_count)
        extermination_indices = argpart[:-self.survivor_count]
        survivor_indices = argpart[-self.survivor_count:]
        for index in extermination_indices:
            parent_a = np.count_nonzero(cumulative_fitness < random())
            parent_b = np.count_nonzero(cumulative_fitness < random())
            crossover_point = randint(0, self.genes_per_specimen - 1)
            # Kinda equivalent to combining the binary strings
            #self.genepool[index,:crossover_point] = self.genepool[parent_a,:crossover_point]
            #self.genepool[index,crossover_point:] = self.genepool[parent_b,crossover_point:]
            #self.genepool[index,crossover_point] += np.random.uniform(-1, 1, 1)[0]
            selector = np.random.random_sample(self.genes_per_specimen) > 0.5

            self.genepool[index, selector] = self.genepool[parent_a, selector]
            self.genepool[index, np.logical_not(selector)] = self.genepool[parent_b, np.logical_not(selector)]
        
        mutation_filter = np.random.uniform(0, 1, self.genepool.shape) < self.mutation_rate
        self.genepool[mutation_filter] += np.random.normal(0, self.mutation_stdev, self.genepool[mutation_filter].shape)
        self.genepool = np.mod(self.genepool + 1, 2) - 1

        if self.elitism:
            self.genepool[best_index] = best_gene_in_generation

    def save(self, filepath: str) -> None:
        np.savetxt(filepath, self.genepool)

    def load(self, filepath: str) -> None:
        data = np.loadtxt(filepath)
        data_pop, data_gen = data.shape
        if data_pop == self.population_count and data_gen == self.genes_per_specimen:
            self.genepool = np.loadtxt(filepath)
        else:
            raise ValueError(
                f"Error loading '{filepath}': "
                f"Data shape ({data_pop}, {data_gen}) does not match up with "
                f"({self.population_count}, {self.genes_per_specimen})"
            )


class GANeuralNets(GAPopulation):
    def __init__(self, population: int, p_architecture: list[Layer], p_scale: float=1, **kwargs):
        self.networks = []
        self.scale = p_scale
        self.architecture = p_architecture
        for i in range(population):
            self.networks.append(Network(p_architecture, (-p_scale, p_scale), (-p_scale, p_scale)))
        
        super().__init__(population, len(self.networks[0].weights), **kwargs)
    
    def update_networks(self) -> None:
        for i in range(self.population_count):
            self.networks[i].weights = self.genepool[i] * self.scale


    def genetic_selection(self, fitness: np.ndarray) -> None:
        super().genetic_selection(fitness)
        self.update_networks()

    def process_inputs(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.zeros((len(inputs), self.networks[0].architecture[-1].size))
        for i, inp in enumerate(inputs):
            outputs[i] = self.networks[i].eval(inp).flatten()
        return outputs
    
    def get_best_network(self) -> Network:
        res = Network(self.architecture, (-self.scale, self.scale), (-self.scale, self.scale))
        res.weights = self.best_gene * self.scale
        return res
    
    def load(self, filepath: str) -> None:
        super().load(filepath)
        self.update_networks()



def test_ga():
    pop = GAPopulation(40, 2)
    pop.mutation_rate = 0.01

    for i in range(200):
        x1 = pop.genepool[:,0] * 512
        x2 = pop.genepool[:,1] * 512
        
        # eggholder
        a = np.sqrt(np.fabs(x2 + x1/2 + 47))
        b = np.sqrt(np.fabs(x1 - (x2 + 47)))
        fitness = (x2 + 47)*np.sin(a) - x1*np.sin(b)

        pop.genetic_selection(fitness)

    print(pop.best_val, pop.best_gene)
    pop.save("saved_networks/test_save.dat")
    pop2 = GAPopulation(40, 2)
    pop2.load("saved_networks/test_save.dat")
    print(np.sum(np.square(pop.genepool - pop2.genepool)))
    
if __name__ == "__main__":
    test_ga()
