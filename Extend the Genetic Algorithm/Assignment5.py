import pandas as pd
import random
import copy
import math
import numpy as np

# Load the dataset
df = pd.read_csv("candy-data.csv")

# Preprocess the data to create profit for each candy type
def preprocess_data(demand, df):
    df["profit"] = ((1 + 1 * (df["winpercent"] / 100)) - (1 * df["pricepercent"])) * demand * (df["winpercent"] / 100)
    overall_dict = {}
    list_of_records = df.to_dict('records')
    for record in list_of_records:
        overall_dict[record['competitorname']] = record
    overall_dict.pop('One dime', None)
    overall_dict.pop('One quarter', None)
    return overall_dict

# Line class representing a single production line
class Line:
    def __init__(self, candy_options, candy_dict, candy_type, already_candies=[]):
        self.candy_dict = candy_dict.copy()
        self.candy_list = []
        self.candy_units = 0
        self.base_candy_limit = 8  # Default max units
        self.candy_type = candy_type
        # Filter candies based on type and exclude candies in already_candies
        self.candy_options = [candy for candy in candy_options if candy_type in candy.lower() and candy not in already_candies]
        if not self.candy_options:
            raise ValueError(f"No candies found for candy type '{candy_type}'")
        
        self.random_init_candies()
        self.prev_candy = None
        self.new_candy = None

    def get_candy_units(self, candy_name):
        return 2 if self.candy_dict[candy_name]["pluribus"] == 1 else 1

    def random_init_candies(self):
        tries = 0
        candy_options = self.candy_options.copy()
        while self.candy_units < self.base_candy_limit and tries < 20:
            tries += 1
            if not candy_options:
                break
            candy_choice = random.choice(candy_options)
            units = self.get_candy_units(candy_choice)
            if self.candy_units + units <= self.base_candy_limit:
                self.candy_units += units
                self.candy_list.append(candy_choice)
                candy_options.remove(candy_choice)

    def mutate_candy(self, other_candies):
        tries = 0
        while tries < 20:
            tries += 1
            available_options = [candy for candy in self.candy_options if candy not in other_candies]
            if not available_options:
                return  # Skip mutation if no valid options

            old_candy_choice = random.choice(self.candy_list)
            new_candy_choice = random.choice(available_options)

            old_units = self.get_candy_units(old_candy_choice)
            new_units = self.get_candy_units(new_candy_choice)

            if (self.candy_units - old_units + new_units) <= self.base_candy_limit:
                self.candy_options.append(old_candy_choice)
                self.candy_units = self.candy_units - old_units + new_units
                self.candy_list.remove(old_candy_choice)
                self.candy_list.append(new_candy_choice)
                self.candy_options.remove(new_candy_choice)
                break

    def calc_total_candy_profit(self, demand_dist, num_samples=5):
        total_profit = 0
        for candy_name in self.candy_list:
            sampled_profits = [
                self.candy_dict[candy_name]["profit"] * max(np.random.normal(demand_dist['mean'], demand_dist['std']), 0)
                for _ in range(num_samples)
            ]
            total_profit += sum(sampled_profits) / num_samples
        return total_profit

# Population class with recombination in next-generation selection
class Population:
    def __init__(self, candy_options, candy_dict, members, top_members):
        self.candy_options = candy_options.copy()
        self.candy_dict = candy_dict.copy()
        self.member_num = members
        self.top_members_num = top_members
        self.tournament_size = 4
        self.mutation_rate = 0.2
        self.members = []
        self.demand_dist = {'mean': 1, 'std': 0.1}

        for i in range(self.member_num):
            chocolate_line = Line(candy_options, candy_dict, "chocolate")
            fruit_line = Line(candy_options, candy_dict, "fruit", already_candies=chocolate_line.candy_list)
            self.members.append((chocolate_line, fruit_line))

    def recombination(self, parent1, parent2):
        # Recombination by swapping half of the candy list from each line
        child_chocolate_list = parent1[0].candy_list[:len(parent1[0].candy_list)//2] + parent2[0].candy_list[len(parent2[0].candy_list)//2:]
        child_fruit_list = parent1[1].candy_list[:len(parent1[1].candy_list)//2] + parent2[1].candy_list[len(parent2[1].candy_list)//2:]

        # Ensure no duplicates between chocolate and fruit lines in the child
        child_chocolate_list = list(set(child_chocolate_list) - set(child_fruit_list))
        child_fruit_list = list(set(child_fruit_list) - set(child_chocolate_list))

        # Create new lines for the child
        child_chocolate_line = Line(self.candy_options, self.candy_dict, "chocolate", already_candies=child_fruit_list)
        child_chocolate_line.candy_list = child_chocolate_list

        child_fruit_line = Line(self.candy_options, self.candy_dict, "fruit", already_candies=child_chocolate_list)
        child_fruit_line.candy_list = child_fruit_list

        return (child_chocolate_line, child_fruit_line)

    def mutate(self):
        mutation_number = math.floor(self.member_num * self.mutation_rate)
        to_mutate = random.sample(self.members, mutation_number)
        for line_pair in to_mutate:
            line_pair[0].mutate_candy(line_pair[1].candy_list)
            line_pair[1].mutate_candy(line_pair[0].candy_list)

    def tournament_selection(self):
        selection_list = random.sample(self.members, self.tournament_size)
        selection_list.sort(key=lambda x: x[0].calc_total_candy_profit(self.demand_dist) +
                            x[1].calc_total_candy_profit(self.demand_dist), reverse=True)
        return copy.deepcopy(selection_list[0])

    def new_generation(self):
        new_generation = []
        for i in range(0, self.member_num, 2):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1 = self.recombination(parent1, parent2)
            child2 = self.recombination(parent2, parent1)
            new_generation.extend([child1, child2])
        self.members = new_generation[:self.member_num]

    def run_generation(self):
        self.mutate()
        self.new_generation()

    def get_top_performance(self):
        self.members.sort(key=lambda x: x[0].calc_total_candy_profit(self.demand_dist) +
                          x[1].calc_total_candy_profit(self.demand_dist), reverse=True)
        top_chocolate_line = self.members[0][0]
        top_fruit_line = self.members[0][1]
        top_profit = top_chocolate_line.calc_total_candy_profit(self.demand_dist) + top_fruit_line.calc_total_candy_profit(self.demand_dist)
        return {
            "top_profit": top_profit,
            "chocolate_candies": top_chocolate_line.candy_list,
            "fruit_candies": top_fruit_line.candy_list
        }

# Initialize candy data and run experiments with different hyperparameters
demand = 796142
candy_dict = preprocess_data(demand, df)
candy_options = list(candy_dict.keys())

experiment_results = []
for config in [(10, 15), (15, 10), (20, 20)]:  # (population_size, generations)
    population_size, generations = config
    print(f"\nRunning experiment with population_size={population_size} and generations={generations}\n")
    population = Population(candy_options, candy_dict, population_size, 3)
    
    for generation in range(generations):
        population.run_generation()
    
    # Get the top performer for this configuration
    top_performance = population.get_top_performance()
    experiment_results.append({
        "config": f"population_size={population_size}, generations={generations}",
        "top_profit": top_performance["top_profit"],
         "chocolate_candies": top_performance["chocolate_candies"],
        "fruit_candies": top_performance["fruit_candies"]
    })

# Display the results of each experiment
print("\n-------- Experiment Results --------")
for result in experiment_results:
    print(f"Configuration: {result['config']}")
    print(f"Top Profit: ${result['top_profit']:.2f}")
    print(f"Chocolate Line Candies: {', '.join(result['chocolate_candies'])}")
    print(f"Fruit Line Candies: {', '.join(result['fruit_candies'])}")
    print("------------------------------------")
