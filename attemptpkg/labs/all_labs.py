def show_lab1():
    code = """
# 1.Fuzzy Logic: Temperature to Fan Speed

def low_temp(x):
    if x <= 20:
        return 1
    elif 20 < x < 30:
        return (30 - x) / 10
    else:
        return 0

def medium_temp(x):
    if 20 < x < 30:
        return (x - 20) / 10
    elif 30 <= x <= 40:
        return (40 - x) / 10
    else:
        return 0

def high_temp(x):
    if x <= 30:
        return 0
    elif 30 < x < 40:
        return (x - 30) / 10
    else:
        return 1

def fuzzy_logic(temp):
  #Fuzzification
    low = low_temp(temp)
    medium = medium_temp(temp)
    high = high_temp(temp)

    print(f"Low membership:    {low}")
    print(f"Medium membership: {medium}")
    print(f"High membership:   {high}")

    numerator = (low * 20) + (medium * 50) + (high * 80)
    denominator = low + medium + high

    if denominator == 0:
        return 0

    # Defuzzification
    speed = numerator / denominator
    return speed

# Input
temp = float(input("Enter temperature: "))
speed = fuzzy_logic(temp)
print(f"\nFinal Fan Speed: {speed:.2f}")

"""
    print(code)
# This file contains all lab programs with show_labX() printers.

def show_lab2():
    code = """
#2 copy Defuzzification using Centroid Method
def alpha_cut(fs, lam):
    result = []
    for k in fs:
        if fs[k] >= lam:
            result.append(k)
    return result
# Mean of Maximum (MOM)
def mom(fs):
    max_val = None
    for k in fs:
        if max_val is None or fs[k] > max_val:
            max_val = fs[k]
    total = 0
    count = 0
    for k in fs:
        if fs[k] == max_val:
            total += k
            count += 1
    return total / count
def cog(fs):
    numerator = 0
    denominator = 0
    for i in fs:
        numerator += i*fs[i]
        denominator += fs[i]
    if denominator == 0:
        return 0
    return numerator / denominator
# Example fuzzy output values
fs1 = {"Low": 0.2, "Medium": 0.7, "High": 0.5}
print("Alpha-cut:", alpha_cut(fs1, 0.5))
fs2 = {1:0.2, 2:0.5, 3:0.8, 4:0.5}
print("MOM:", mom(fs2))
print("COG:", cog(fs2))

"""
    print(code)

def show_lab3():
    code = """
#3.ACO
import random
 
# Distance matrix
dist = [
    [0, 12,  5, 20,  8],
    [12,  0, 15,  3, 18],
    [ 5, 15,  0,  9, 14],
    [20,  3,  9,  0,  7],
    [ 8, 18, 14,  7,  0]
]
n = len(dist)
 
# Parameters
ANTS, ITERS, ALPHA, BETA, EVAP = 5, 10, 1, 2, 0.5
 
# Pheromone matrix
pher = [[1.0] * n for _ in range(n)]
 
def next_city(cur, visited):
    probs = []
    for j in range(n):
        if j not in visited:
            tau = pher[cur][j] ** ALPHA
            eta = (1 / dist[cur][j]) ** BETA if dist[cur][j] else 0
            probs.append((j, tau * eta))
    total = sum(p for _, p in probs)
    if not total:
        return random.choice([c for c in range(n) if c not in visited])
    r, s = random.uniform(0, total), 0
    for city, p in probs:
        s += p
        if s >= r:
            return city
 
def path_len(path):
    return sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))
 
best_path, best_len = None, float('inf')
 
for t in range(ITERS):
    all_paths = []
    for _ in range(ANTS):
        path = [random.randint(0, n-1)]
        visited = set(path)
        while len(path) < n:
            path.append(next_city(path[-1], visited))
            visited.add(path[-1])
        if len(path) == n:
            l = path_len(path)
            all_paths.append((path, l))
            if l < best_len:
                best_len, best_path = l, path
 
    # Evaporate
    for i in range(n):
        for j in range(n):
            pher[i][j] *= (1 - EVAP)
 
    # Deposit
    for path, l in all_paths:
        if l:
            for i in range(len(path)-1):
                pher[path[i]][path[i+1]] += 1 / l
 
    print(f"Iter {t+1:2d} | Best: {' -> '.join(map(str, best_path))} | Cost: {best_len}")
 
print("=" * 45)
print(f"Best Path : {' -> '.join(map(str, best_path))}")
print(f"Best Cost : {best_len}")
"""
    print(code)


def show_lab4():
    code = """
import random #4.PSO

# Objective function (minimize)
def fitness(x):
    return x**2

# Parameters
num_particles = 5
iterations = 10
w = 0.5       # inertia
c1 = 1        # cognitive
c2 = 2        # social

# Initialize particles
particles = [random.uniform(-10, 10) for _ in range(num_particles)]
velocities = [random.uniform(-1, 1) for _ in range(num_particles)]
pBest = particles[:]
gBest = min(particles, key=fitness)

for i in range(iterations):
    for j in range(num_particles):
        r1 = random.random()
        r2 = random.random()

        # Update velocity
        velocities[j] = (w * velocities[j] +
                         c1 * r1 * (pBest[j] - particles[j]) +
                         c2 * r2 * (gBest - particles[j]))

        # Update position
        particles[j] += velocities[j]

        # Update Pbest
        if fitness(particles[j]) < fitness(pBest[j]):
            pBest[j] = particles[j]

    # Update Gbest
    gBest = min(pBest, key=fitness)
    print(f"Iteration {i+1}: gBest = {round(gBest, 4)}, f(x) = {round(fitness(gBest), 6)}")

print("\nBest Position (Solution):", round(gBest, 4))
print("Minimum Value:", round(fitness(gBest), 4))
"""
    print(code)

def show_lab5():
    code = """
import random #5. Genetic Algo

# Fitness function (maximize)
def fitness(x):
    return x**3

# Parameters
population_size = 6
generations = 20
mutation_rate = 0.1

# Initialize population (random integers)
population = [random.randint(0, 10) for _ in range(population_size)]

for gen in range(generations):
    # Fitness Evaluation
    population = sorted(population, key=fitness, reverse=True)
    print(f"Generation {gen+1}: {population}")

    # Selection (top 2)
    parent1, parent2 = population[0], population[1]

    # Crossover (simple average)
    child = (parent1 + parent2) // 2

    # Mutation
    if random.random() < mutation_rate:
        child += random.randint(-2, 2)

    # Replace worst individual
    population[-1] = child

# Final result
best = max(population, key=fitness)

print("Best Solution:", best)
print("Maximum Value:", fitness(best))

"""
    print(code)

def show_lab6():
    code = """
import random #6. Grey wolf
 
def fitness(x):
    return x ** 2
 
# Parameters
WOLVES, ITERS = 5, 20
 
wolves = [random.uniform(-10, 10) for _ in range(WOLVES)]
 
for t in range(ITERS):
    wolves.sort(key=fitness)
    alpha, beta, delta = wolves[0], wolves[1], wolves[2]
 
    a = 2 - t * (2 / ITERS)  # decreases from 2 to 0
 
    new_wolves = []
    for w in wolves:
        X_new = 0
        for leader in [alpha, beta, delta]:
            r1, r2 = random.random(), random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            D = abs(C * leader - w)
            X_new += leader - A * D
        new_wolves.append(X_new / 3)
 
    wolves = new_wolves
    print(f"Iter {t+1:2d} | Alpha: {round(alpha,4):8.4f} | Best f(x): {round(fitness(alpha),6):.6f}")
 
print("=" * 45)
print(f"Best Position : {round(alpha, 4)}")
print(f"Minimum Value : {round(fitness(alpha), 6)}")
"""
    print(code)

def show_lab7():
    code = """
import numpy as np #IWD

n = 5
soil = np.ones((n, n))

# Distance matrix (example)
dist = np.random.randint(1, 10, (n, n))

def select_next(current, visited):
    probs = [1/soil[current][j] if j not in visited else 0 for j in range(n)]
    probs = np.array(probs) / sum(probs)
    return int(np.random.choice(range(n), p=probs))

best_path, best_cost = [], float('inf')

for it in range(10):
    print(f"\nIter {it+1}")
    for i in range(n):
        visited, current = [i], i
        cost = 0

        while len(visited) < n:
            nxt = select_next(current, visited)
            cost += dist[current][nxt]   # real cost
            soil[current][nxt] += 0.1
            visited.append(nxt)
            current = nxt

        print(f"Start {i}: {visited} | Cost: {cost}")

        if cost < best_cost:
            best_cost, best_path = cost, visited

    print(f"Best so far: {best_path} | Cost: {best_cost}")

print("\nFinal Best Path:", best_path)
print("Final Cost:", best_cost)

"""
    print(code)

def show_lab8():
    code = """
import random, math #8.Firefly
 
def fitness(x):
    return x**2 - 10 * math.cos(2 * math.pi * x) + 10
 
# Parameters
N, ITERS = 10, 20
ALPHA, BETA0, GAMMA = 0.5, 1, 0.5  # more randomness, less absorption = more exploration
 
fireflies = [random.uniform(-10, 10) for _ in range(N)]
 
global_best = min(fireflies, key=fitness)
 
for t in range(ITERS):
    for i in range(N):
        for j in range(N):
            if fitness(fireflies[j]) < fitness(fireflies[i]):
                r = abs(fireflies[i] - fireflies[j])
                beta = BETA0 * math.exp(-GAMMA * r**2)
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + ALPHA * (random.random() - 0.5)
 
    curr_best = min(fireflies, key=fitness)
    if fitness(curr_best) < fitness(global_best):
        global_best = curr_best
 
    print(f"Iter {t+1:2d} | Curr x: {round(curr_best,4):8.4f} | f(x): {round(fitness(curr_best),4):7.4f} | Global Best f(x): {round(fitness(global_best),4):.4f}")
 
print("=" * 55)
print(f"Best Position : {round(global_best, 4)}")
print(f"Minimum Value : {round(fitness(global_best), 4)}")

"""
    print(code)


def show_lab9():
    code = """
import random

# 9. ARTIFICIAL BEE COLONY

def fitness(x):
    return x**2  # Objective function: minimize x^2

# Initialize random solutions (food sources)
solutions = [random.uniform(-10, 10) for _ in range(5)]

for t in range(20):  # Number of iterations (cycles)
    for i in range(len(solutions)):
        # Select a random neighbor solution
        k = random.randint(0, len(solutions) - 1)
        
        # Generate a new candidate solution
        phi = random.uniform(-1, 1)
        new_solution = solutions[i] + phi * (solutions[i] - solutions[k])
        
        # Greedy selection: keep the better solution
        if fitness(new_solution) < fitness(solutions[i]):
            solutions[i] = new_solution

    # Find best solution in current population
    best = min(solutions, key=fitness)

    print(f"[ABC] Iter {t+1:2d} | Best position: {best:.4f} | Minimum value: {fitness(best):.4f}")

# Final result
print(f"\nABC Result: Best position = {best:.4f}, Minimum value = {fitness(best):.4f}")




"""
    print(code)


def show_lab10():
    code = """
#program 10
!pip install squarify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

df = pd.read_csv("Dataset_10.csv")

# Use only the first listed position
df['MainPos'] = df['player_positions'].str.split(',').str[0]

# 1. Bar chart: number of players per position
df['MainPos'].value_counts().plot(kind='bar')
plt.title("Players per Position"); plt.ylabel("Count")
plt.show()

# 2. Donut chart: rating distribution by position
bins = [0,60,70,80,90,100]
df['RatingRange'] = pd.cut(df['overall'], bins)
sizes = df['RatingRange'].value_counts()
plt.pie(sizes, labels=sizes.index, autopct='%1.1f%%')
plt.title("Rating Distribution")
plt.gca().add_artist(plt.Circle((0,0),0.70,color='white'))
plt.show()

# 3. Treemap (tree diagram): overall rating grouped by position
data = df.groupby('MainPos')['overall'].sum()
squarify.plot(sizes=data.values, label=data.index)
plt.title("Treemap: Total Rating by Position")
plt.axis('off')
plt.show()

# 4. Interpretation
print('
Bar Chart : Shows which positions have the most players.
Donut Chart : Shows how overall ratings are distributed across ranges.
Treemap : Shows the hierarchical structure: which positions contribute most
           to total rating across the dataset.
')
"""
    print(code)


