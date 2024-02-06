import random
import numpy as np
import pandas as pd
from random import randint, shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import geometric_mean_score
from scipy.spatial.distance import hamming

class EUS_CHC(object):
    def __init__(self, n_population, n_generation):
        self.n_population = n_population
        self.n_generation = n_generation

    # def size_of_chromosome(self):
    #     if self.n_minor < 500:
    #         size = self.n_minor+int(0.2*self.n_minor)
    #     else:
    #         size = self.n_minor+int(0.15*self.n_minor)
    #     return size

    def create_chromosome(self):
        chromosome = np.full(self.n_major, 0)
        idx_change = np.random.randint(self.n_major, size=self.n_minor+int(0.15*self.n_minor))
        chromosome[idx_change] = 1
        return list(chromosome)

    def create_population(self):
        pop = [self.create_chromosome() for x in range(self.n_population)]
        return pop

    def fitness_eus(self, chromosome, majority_data, minority_data, iboost, best_chromosomes):
        # data majority yang diselect oleh kromosom
        df_major = majority_data[np.array(chromosome,dtype=bool)]
        # data minority
        df_minor = minority_data
        # gabung kedua data
        samples = pd.concat([df_major, df_minor], ignore_index=True)
        samples_X = samples.iloc[:,:-1]
        samples_y = samples.iloc[:,-1]
        samples_X_train, samples_X_test, samples_y_train, samples_y_test = train_test_split(samples_X, samples_y, test_size=0.3, random_state=42)
        # fit dengan 1NN
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(samples_X_train, samples_y_train)
        # predict
        y_pred_neigh = neigh.predict(samples_X_test)
        # dapatkan nilai Geometric meannya
        GM = geometric_mean_score(samples_y_test, y_pred_neigh)
        # hitung fitness
        P = 0.2
        if df_major.shape[0] > 0:
            fitness = GM - (np.abs(1-((df_minor.shape[0]/df_major.shape[0])*P)))
        else:
            fitness = GM - P
        if iboost > 0:
            N = self.n_minor + self.n_major
            B = (N - iboost - 1) / N
            IR = self.n_minor/self.n_major
            H = []
            for i in best_chromosomes:
                h = hamming(chromosome, i) * len(chromosome)
                H.append(h)
            fitness = (fitness * (1/B) * (10/IR)) + ((min(H)/self.n_minor)*B)
        return fitness

    def hux(self, ind1, ind2):
        child_one = []
        child_two = []
        hamming_dist = hamming(ind1, ind2) * len(ind1)
        nr_swaps = 0
        for x in range(0, len(ind1)):
            if (ind1[x] == ind2[x]) or (random.random() > 0.25) or (nr_swaps > hamming_dist / 2):
                #same, just copy to both
                child_one.append(ind1[x])
                child_two.append(ind2[x])
            else:
                #different, swap with .25 probability, until hamming swaps
                nr_swaps += 1
                child_one.append(ind2[x])
                child_two.append(ind1[x])
        return [child_one,child_two]

    def crossover(self, parents, threshold):
        childs = []
        if len(parents) % 2 == 0:
            for i in range(0,len(parents),2):
                p1 = parents[i]
                p2 = parents[i+1]
                hamming_distance = hamming(p1, p2) * len(p1)
                if hamming_distance > threshold:
                    # recombine
                    children = self.hux(p1, p2)
                    childs.append(children[0])
                    childs.append(children[1])
        else:
            for i in range(0,len(parents)-1,2):
                p1 = parents[i]
                p2 = parents[i+1]
                hamming_distance = hamming(p1, p2) * len(p1)
                if hamming_distance > threshold:
                    # recombine
                    children = self.hux(p1, p2)
                    childs.append(children[0])
                    childs.append(children[1])
        return childs

    def elitisme(self, population, all_fitness, n_child):
        zipped = list(zip(population, all_fitness))
        df = pd.DataFrame(zipped, columns = ['chromosome', 'fitness'])
        df = df.sort_values(by=['fitness'], ascending=False)
        if df.duplicated(subset='fitness').sum() > n_child:
            df = df.iloc[:self.n_population,:]
        else:
            df = df.drop_duplicates(subset='fitness', keep="first")
            df = df.iloc[:self.n_population,:]
        records = df.to_records(index=False)
        result = list(zip(*list(records)))
        return result[0], result[1]

    def reinitialized(self, population, fitness):
        idx_best = fitness.index(max(fitness)) 
        best = population[idx_best]
        new = [best]
        for x in range(1, self.n_population):
            next_gen = best[:]
            for i in range(0, len(next_gen)):
                if 0.35 > random.random():
                    next_gen[i] = randint(0, 1)
            new.append(next_gen)
        return new

    def under_sampling(self, majority_data, minority_data, iboost, best_chromosomes):
        self.n_major = majority_data.shape[0]
        self.n_minor = minority_data.shape[0]

        t = 0
        threshold = self.n_major/4
        init_pop = self.create_population()

        while t < self.n_generation:
            # select parents (all population beecome parents but in random order)
            shuffle(init_pop)
            # crossover 
            childs_population = self.crossover(init_pop, threshold)
            # select best (parent population, childs_population)
            tmp = init_pop + childs_population
            all_fitness = [self.fitness_eus(tmp[i], majority_data, minority_data, iboost, best_chromosomes) for i in range(len(tmp))]
            new_pop, fitness = self.elitisme(tmp, all_fitness, len(childs_population))

            if len(childs_population) == 0:
                threshold -= 1
            
            if threshold < 0:
                new_pop = self.reinitialized(new_pop, fitness)
                threshold = self.n_major/4
            
            init_pop = list(new_pop)
            t += 1
        
        idx_best_ch = fitness.index(max(fitness))
        best_chromosome = init_pop[idx_best_ch]
        return best_chromosome 