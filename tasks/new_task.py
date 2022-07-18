import os
import random
import math
from enum import Enum
import modneat
from modneat import visualize
from scipy.spatial import distance

global max_vector_distance 

MAX_STEP = 100

class timing_random:
    """
    ルール更新ステップ数は毎回ランダムに決定される
    """
    def __init__(self, network_type):
        self.network_type = network_type
        self.change_timings = [] #空ならば毎回ランダムにルール変更ステップを決定。空でなければ、
                                 #各試行の最初でリストの要素中からランダムにタイミングを決定
        self.desired_outputs = [0.0, 1.0] #change_timingが来ると一つ次の要素がdesired_outputになる。
        self.desired_pointer = 0

    def update_desired_output(self):
        print(' DESIRED_UPDATED ')
        self.desired_pointer += 1
        if(self.change_timing == None):
            self.current_change_timing = random.choice([5,6,7,8,9,10])
        else:
            self.current_change_timing = self.change_timing
        print(self.current_change_timing)

    def eval_fitness(self, net):
        if(self.change_timings == []):
            self.change_timing = None
            self.current_change_timing = random.choice([5,6,7,8,9,10])
        else:
            self.change_timing = random.choice(self.change_timings)
            self.current_change_timing = self.change_timing

        step = 0
        current_rule_step = 0
        fitness = 0.0

        while(step < MAX_STEP):
            step += 1
            current_rule_step += 1
            print(f'*** step {step} ***')
            output = net.activate([1, 0, 0])
            desired_output = self.desired_outputs[self.desired_pointer % len(self.desired_outputs) ]
            print(f'desired: {desired_output}')
            loss = abs(output[0] - desired_output)
            fitness += (1.0 - loss)

            output = net.activate([0, 1, loss])

            if(current_rule_step % self.current_change_timing == 0):
                self.update_desired_output()
                current_rule_step = 0

        return fitness / MAX_STEP

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness = self.eval_fitness(net)
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Visualize the experiment results
        node_names = {-1:'get_output_phase', -2: 'feedback_phase', -3: 'loss', 0:'output'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class timing_static(timing_random):
    """
    ルール更新ステップ数は毎回10ステップ毎
    """
    def __init__(self, network_type):
        self.network_type = network_type
        self.change_timings = [10] #空ならば毎回ランダムにルール変更ステップを決定。空でなければ、
                                 #各試行の最初でリストの要素中からランダムにタイミングを決定
        if(self.change_timings == []):
            self.change_timing = None
            self.current_change_timing = random.choice([5,6,7,8,9,10])
        else:
            self.change_timing = random.choice(self.change_timings)
            self.current_change_timing = self.change_timing
        self.desired_outputs = [0.0, 1.0] #change_timingが来ると一つ次の要素がdesired_outputになる。
        self.desired_pointer = 0

class timing_random_generation(timing_random):
    """
    ルール更新ステップ数は各試行でランダムに決定
    """
    def __init__(self, network_type):
        self.network_type = network_type
        self.change_timings = [5,6,7,8,9,10] #空ならば毎回ランダムにルール変更ステップを決定。空でなければ、
                                 #各試行の最初でリストの要素中からランダムにタイミングを決定
        self.desired_outputs = [0.0, 1.0] #change_timingが来ると一つ次の要素がdesired_outputになる。
        self.desired_pointer = 0

    def eval_fitness(self, net):

        fitness = 0.0

        for timing in self.change_timings:
            print('実験開始')
            net.reset()
            step = 0 #ここ確認
            current_rule_step = 0
            self.change_timing = timing
            self.current_change_timing = self.change_timing

            while(step < MAX_STEP):
                step += 1
                current_rule_step += 1
                print(f'*** step {step} ***')
                output = net.activate([1, 0, 0])
                desired_output = self.desired_outputs[self.desired_pointer % len(self.desired_outputs) ]
                print(f'desired: {desired_output}')
                loss = abs(output[0] - desired_output)
                fitness += (1.0 - loss)

                output = net.activate([0, 1, loss])

                if(current_rule_step % self.current_change_timing == 0):
                    self.update_desired_output()
                    current_rule_step = 0

        return fitness / (MAX_STEP * len(self.change_timings))

class changing_random:
    """
    次のdesired_outputは毎回ランダムに決定(増加 or 減少)
    """
    def __init__(self, network_type):
        self.network_type = network_type
        self.desired_outputs = [0.0, 0.333, 0.666, 1.0]
        self.desired_output_pointer = 0

    def eval_fitness(self, net):
        fitness = 0.0
        step = 0
        while(step < MAX_STEP):
            step += 1

            print(f'step: {step}')

            output = net.activate([1, 0, 0])
            desired_output = self.desired_outputs[self.desired_output_pointer]
            print(f'desired_output: {desired_output}')
            loss = abs(output[0] - desired_output)
            fitness += (1.0 - loss)

            output = net.activate([0, 1, loss])
            
            if(step %10 == 0):
                self.update_desire_pointer(random.choice([-1, 1]))

        return fitness / MAX_STEP
    
    def update_desire_pointer(self, shift):
        print('***ルール変更しました***')
        self.desired_output_pointer += shift
        if(self.desired_output_pointer < 0):
            self.desired_output_pointer = len(self.desired_outputs) -1
        elif(self.desired_output_pointer >= len(self.desired_outputs)):
            self.desired_output_pointer = 0

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness = self.eval_fitness(net)

    def show_results(self, best_genome, config, stats, out_dir):
        # Visualize the experiment results
        node_names = {-1:'get_output_phase', -2: 'feedback_phase', -3: 'loss', 0:'output'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class changing_static(changing_random):

    def eval_fitness(self, net):
        fitness = 0.0
        step = 0
        while(step < MAX_STEP):
            step += 1
            print(f'step: {step}')
            output = net.activate([1, 0, 0])
            desired_output = self.desired_outputs[self.desired_output_pointer]
            print(f'desired_output: {desired_output}')
            loss = abs(output[0] - desired_output)
            fitness += (1.0 - loss)

            output = net.activate([0, 1, loss])
            
            if(step %10 == 0):
                self.update_desire_pointer(1)

        return fitness / MAX_STEP

class changing_random_generation(changing_random):

    def eval_fitness(self, net):
        fitness = 0.0
        patterns = [-1, 1]
        print('実験開始')

        for p in patterns:
            print(f'パターン:{p}')
            net.reset()
            self.desired_output_pointer = 0
            step = 0
            while(step < MAX_STEP):
                step += 1
                print(f'step is {step}')

                output = net.activate([1, 0, 0])
                desired_output = self.desired_outputs[self.desired_output_pointer]
                print(f'desired_output: {desired_output}')
                loss = abs(output[0] - desired_output)
                fitness += (1.0 - loss)

                output = net.activate([0, 1, loss])
                
                if(step %10 == 0):
                    self.update_desire_pointer(p)

        return fitness / (MAX_STEP * 2)

class xor:
    # The XOR inputs and expected corresponding outputs for fitness evaluation
    def __init__(self, network_type):
        self.network_type = network_type
        self.xor_inputs  = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        self.xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

    def eval_fitness(self, net):
        """
        Arguments:
            net: The feed-forward neural network generated from genome
        Returns:
            The fitness score - the higher score the means the better 
            fit organism. Maximal score: 16.0
        """
        error_sum = 0.0
        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = net.activate(xi)
            error_sum += abs(output[0] - xo[0])
        # Calculate amplified fitness
        fitness = (4 - error_sum) ** 2
        return fitness

    def eval_genomes(self, genomes, config):
        """
        Arguments:
            genomes: The list of genomes from population in the 
                    current generation
            config: The configuration settings with algorithm
                    hyper-parameters
        """
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = self.network_type.create(genome, config)
            genome.fitness = self.eval_fitness(net)
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Display the best genome among generations.
        print('\n ************************* Finish evolution *************************  \n')
        print('\nBest genome:\n{!s}'.format(best_genome))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        net = self.network_type.create(best_genome, config)
        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

        # Check if the best genome is an adequate XOR solver
        best_genome_fitness = self.eval_fitness(net)
        if best_genome_fitness > config.fitness_threshold:
            print("\n\nSUCCESS: The XOR problem solver found!!!")
        else:
            print("\n\nFAILURE: Failed to find XOR problem solver!!!")

        # Visualize the experiment results
        node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class non_static(xor):
    # The XOR inputs and expected corresponding outputs for fitness evaluation
    def __init__(self, network_type):
        self.network_type = network_type
        self.xor_inputs  = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        self.xor_outputs = [   (1.0,),     (0.66,),     (0.33,),     (0.0,)]

class dummy_net:
    def activate(self, input_list):
        return [random.random()]

    def reset(self):
        pass

if __name__=='__main__':
    t = timing_random_generation(network_type=dummy_net)
    n = dummy_net()
    fitness = t.eval_fitness(n)
    print(fitness)