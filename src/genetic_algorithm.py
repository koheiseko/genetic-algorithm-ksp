import random
import numpy as np
from prettytable import PrettyTable

class GeneticAlgorithm():
    def __init__(self, weights, vals, max_weight, generations, population_size, mutation_rate):

        self.weights = weights
        self.vals = vals
        self.max_weight = max_weight
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.table = PrettyTable(['Generation', 'Standard deviation', 'Best individual', 'Best fitness', 'Worst fitness'])

        self.best_global_fitness = 0
        self.best_global_individual = 0

    def _create_population(self, population_size):
        population = np.array([[random.randint(0, 1) for _ in range(len(self.vals))] for _ in range(population_size)])
        
        return population
    
    def _get_fitness(self, individual):
        total_weight = 0
        total_vals = 0
        p = 0

        for gene, v, w in zip(individual, self.vals, self.weights): 
            
            total_vals += gene * v
            total_weight += gene * w
            if v / w > p:
                p = v / w

        if total_weight <= self.max_weight:
            fitness = total_vals - 0
        else:
            pen = 10 * p * (total_weight - self.max_weight)
            fitness = total_vals - pen     
        
        return fitness
    
    def _selection(self, population, fitnesses, tournament_size=2):
        selected = []

        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]

            selected.append(winner)

        return np.array(selected)
    
    def _crossover(self, parent_1, parent_2):

        alpha = np.random.randint(0, 8)

        children_1 = np.concatenate((parent_1[:alpha], parent_2[alpha:]))
        children_2 = np.concatenate((parent_2[:alpha], parent_1[alpha:]))

        return children_1, children_2
    
    def _mutation(self, individual, mutation_rate):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                match individual[i]:
                    case 1:
                        individual[i] = 0
                    case 0:
                        individual[i] = 1
        return individual
    
    def execution(self):
        population = self._create_population(self.population_size)
        self.table.clear_rows()
        
        for generation in range(self.generations):
            fitnesses = np.array([self._get_fitness(individual) for individual in population])

            mean = np.sum(fitnesses) / len(fitnesses)
            standard_deviation = np.sqrt((np.sum((fitnesses - mean) ** 2)) / len(fitnesses))

            best_individual = max(population, key=self._get_fitness)
            worst_individual = min(population, key=self._get_fitness)
            best_fitness = self._get_fitness(best_individual)

            if best_fitness > self.best_global_fitness:
                self.best_global_fitness = best_fitness
                self.best_global_individual = best_individual

            worst_fitness = self._get_fitness(worst_individual)

            self.table.add_row([generation + 1, standard_deviation, best_individual, best_fitness, worst_fitness])

            population = self._selection(population, fitnesses)

            next_population = []

            for i in range(0, len(population), 2):
                parent_1 = population[i]
                parent_2 = population[i + 1]

                children_1, children_2 = self._crossover(parent_1, parent_2)

                next_population.append(self._mutation(children_1, self.mutation_rate))
                next_population.append(self._mutation(children_2, self.mutation_rate))
            
            next_population[0] = best_individual
            population = np.array(next_population)

        return best_individual, best_fitness