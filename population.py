from __future__ import annotations

from networks import FFNN, Layer

from random import randint, random
import numpy as np
from math import sin, cos, sqrt, fabs
import os.path

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
            #parent_a = np.count_nonzero(cumulative_fitness < random())
            #parent_b = np.count_nonzero(cumulative_fitness < random())
            parent_a = survivor_indices[randint(0, self.survivor_count-1)]
            parent_b = survivor_indices[randint(0, self.survivor_count-1)]
            
            # Kinda equivalent to combining the binary strings
            #crossover_point = randint(0, self.genes_per_specimen - 1)
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
        np.savetxt(os.path.join("saved_networks", filepath), self.genepool)

    def load(self, filepath: str) -> None:
        data = np.loadtxt(os.path.join("saved_networks", filepath))
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
    def __init__(self, population: int, p_architecture: list[Layer], **kwargs):
        self.weight_scale = kwargs.get("weight_scale", 1)
        self.bias_scale = kwargs.get("bias_scale", self.weight_scale)
        self.extra_genomes_count = kwargs.get("extra_genomes", 0)
        self.architecture = p_architecture
        self.networks = []
        for i in range(population):
            self.networks.append(FFNN(p_architecture, 
                (-self.weight_scale, self.weight_scale), (-self.bias_scale, self.bias_scale)))
        
        super().__init__(population, len(self.networks[0].weights) + self.extra_genomes_count, **kwargs)
        self.filter = self.genepool[:,0] == self.genepool[:,0]
    
    def update_networks(self) -> None:
        for i in range(self.population_count):
            self.networks[i].weights = self.genepool[i] * self.weight_scale

    def genetic_selection(self, fitness: np.ndarray) -> None:
        super().genetic_selection(fitness)
        self.update_networks()

    def process_inputs(self, inputs: np.ndarray) -> np.ndarray:
        outputs = np.zeros((len(inputs), self.networks[0].architecture[-1].size))
        input_index = 0
        for i, net in enumerate(self.networks):
            if self.filter[i]:
                outputs[input_index] = net.eval(inputs[input_index]).flatten()
                input_index += 1
        return outputs
    
    def get_best_network(self) -> FFNN:
        res = FFNN(self.architecture, (-self.weight_scale, self.weight_scale), (-self.weight_scale, self.weight_scale))
        res.weights = self.best_gene * self.weight_scale
        return res
    
    def load(self, filepath: str) -> None:
        super().load(filepath)
        self.update_networks()

    def elite_sample(self, population: int) -> GANeuralNets:
        res = GANeuralNets(population, self.architecture, 
                           weight_scale=self.weight_scale, bias_scale=self.bias_scale)
        for i in range(population):
            res.networks[i] = self.get_best_network()
        return res
    
    @property
    def extra_genomes(self) -> np.ndarray:
        return self.genepool[self.filter,-self.extra_genomes_count:]


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
    pop.save("test_save.dat")
    pop2 = GAPopulation(40, 2)
    pop2.load("test_save.dat")
    print(np.sum(np.square(pop.genepool - pop2.genepool)))
    
if __name__ == "__main__":
    test_ga()
