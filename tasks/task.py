import os
import random
import modneat
from modneat import visualize
from scipy.spatial import distance

global max_vector_distance 
max_vector_distance = 2.44949

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


class three_rules_random:
    def __init__(self, network_type, test = False):
        self.network_type = network_type
        self.current_rule = random.randint(0,2)
        self.test = test
        if(self.test):
            print(self.__class__)

    def change_rule(self):
        while True:
            next_rule = random.randint(0,2)
            if(next_rule != self.current_rule):
                if(self.test):
                    print('Rule changed from %d to %d' % (self.current_rule, next_rule))
                self.current_rule = next_rule
                break

    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = random.randint(5,10)
        fitness = 0

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step %d *** ' %step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[self.current_rule] = 1
            output = net.activate([1, 0, 0])
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance 
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error])

            if(rule_change_timing == current_rule_step):
                current_rule_step = 0
                rule_change_timing = random.randint(5,10)
                self.change_rule()
        
        return fitness / 100

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            net = self.network_type.create(genome, config)
            genome.fitness = self.eval_fitness(net)

    def show_results(self, best_genome, config, stats, out_dir):
        # Visualize the experiment results
        node_names = {-1:'get_output_flag', -2: 'feedback_flag', -3: 'previous_error', 0:'output_1', 1:'output_2', 2:'output_3'}
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class three_rules_random_feedback(three_rules_random):
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = random.randint(5,10)
        fitness = 0

        previous_output = [0,0,0]

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step %d *** ' %step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[self.current_rule] = 1
            output = net.activate([1, 0, 0] + previous_output)
            previous_output = output
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance 
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error] + previous_output)
            previous_output = output

            if(rule_change_timing == current_rule_step):
                current_rule_step = 0
                rule_change_timing = random.randint(5,10)
                self.change_rule()
        
        return fitness / 100


class three_rules_cyclic(three_rules_random):
    def __init__(self, network_type, test=False):
        self.network_type = network_type
        self.current_rule = 0
        self.test = test
        if(self.test):
            print(self.__class__)

    def change_rule(self):
        if(self.test):
            print('*** Rule changed ***')
            print(self.current_rule)
        self.current_rule += 1
        self.current_rule %= 3
        if(self.test):
            print(self.current_rule)

class three_rules_cyclic_feedback(three_rules_cyclic):
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = random.randint(5,10)
        fitness = 0

        previous_output = [0,0,0]

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step %d *** ' %step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[self.current_rule] = 1
            output = net.activate([1, 0, 0] + previous_output)
            previous_output = output
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance 
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error] + previous_output)
            previous_output = output

            if(rule_change_timing == current_rule_step):
                current_rule_step = 0
                rule_change_timing = random.randint(5,10)
                self.change_rule()
        
        return fitness / 100


class three_rules_vary_cyclic(three_rules_random):
    def __init__(self, network_type, test=False):
        self.network_type = network_type
        self.current_rule = 0
        self.test = test
        if(self.test):
            print(self.__class__)

    def change_rule(self):
        if(self.test):
            print('*** Rule changed ***')
            print(self.current_rule)
        self.current_rule += 1
        self.current_rule %= 3
        if(self.test):
            print(self.current_rule)
            
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = random.randint(5,10)
        fitness = 0

        self.rule_list = [0, 1, 2]
        random.shuffle(self.rule_list)

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step %d *** ' %step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[ self.rule_list[self.current_rule] ] = 1
            output = net.activate([1, 0, 0])
            if(self.test):
                print('required_output is : ', required_output)
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error])

            if(rule_change_timing == current_rule_step):
                current_rule_step = 0
                rule_change_timing = random.randint(5,10)
                self.change_rule()
        
        return fitness / 100

class three_rules_vary_cyclic_feedback(three_rules_vary_cyclic):
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = random.randint(5,10)
        fitness = 0

        self.rule_list = [0, 1, 2]
        random.shuffle(self.rule_list)

        previous_output = [0,0,0]

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step %d *** ' %step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[ self.rule_list[self.current_rule] ] = 1
            output = net.activate([1, 0, 0] + previous_output)
            previous_output = output
            if(self.test):
                print('required_output is : ', required_output)
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error] + previous_output)
            previous_output = output

            if(rule_change_timing == current_rule_step):
                current_rule_step = 0
                rule_change_timing = random.randint(5,10)
                self.change_rule()
        
        return fitness / 100

class three_rules_static(three_rules_cyclic):
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = 5
        fitness = 0

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step ***:', step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[self.current_rule] = 1
            output = net.activate([1, 0, 0])
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error])

            if(current_rule_step == rule_change_timing):
                current_rule_step = 0
                self.change_rule()

        return fitness / 100

class three_rules_static_feedback(three_rules_static):
    def eval_fitness(self, net):
        step = 0
        current_rule_step = 0
        rule_change_timing = 5
        fitness = 0

        previous_output = [0,0,0]

        for step in range(100):
            step += 1
            if(self.test):
                print('*** step ***:', step)
            current_rule_step += 1

            required_output = [0, 0, 0]
            required_output[self.current_rule] = 1
            output = net.activate([1, 0, 0] + previous_output)
            previous_output = output
            dist = distance.euclidean(output, required_output)
            assert(0 <= dist <= max_vector_distance), ('required_output:', required_output, 'output:', output, 'dist:', dist)
            if(dist != 0.0):
                error = dist / max_vector_distance
            else:
                error = dist
            fitness += (1.0 - error)

            output = net.activate([0, 1, error] + previous_output)
            previous_output = output

            if(current_rule_step == rule_change_timing):
                current_rule_step = 0
                self.change_rule()

        return fitness / 100


class dummy_net:
    def activate(self, input):
        return [0,0,0]

if __name__=='__main__':
    n = dummy_net()
    t = three_rules_static(0)
    t.eval_fitness(n)
