from modneat import visualize
import numpy
import myenvs.myenvs as myenvs
import gym
import os

class single_cyclic_task:
    def __init__(self, network_type, cycle = 10, cycle_cnt_max = 10, action_num = 2, noise = 0.0):
        self.network_type = network_type
        self.cycle = cycle
        self.cycle_cnt_max = cycle_cnt_max
        self.action_num = action_num
        self.noise = noise #ここに指定した値の割合で、ルール変更が発生しない
    
    def eval_fitness(self, net):
        self.E = gym.make('static_cyclic_env-v0', cycle = self.cycle, cycle_cnt_max = self.cycle_cnt_max, action_num = self.action_num, noise=self.noise)
        normal_fitness = 0.0
        bonus_fitness = 0.0
        history = {'fb1_hist':[], 'fb2_hist':[], 'success_hist':[], 'bonus_hist':[], 'fitness_hist':[]}
        net.reset()
        assert len(net.output_nodes) == self.action_num, 'Error, モデルのoutput_node数とタスクのaction_numが一致しません。'
        observation = self.E.reset()
        while(True):
            #get output phase
            input_vector = [1.0, observation, 0.0, 0.0] #bias_input, observation, normal_feedbak, bonus_feedback
            action = numpy.argmax(net.activate(input_vector))
            observation, reward, done, info = self.E.step(action)

            if(done == True):
                break

            if(reward == 1.0):
                normal_fitness += 1.0
                feedback1 = 1.0
                if(info['is_bonus']): 
                    #ボーナス時に成功
                    bonus_fitness += 1.0
                    feedback2 = 1.0
                else: 
                    #ボーナス時に失敗
                    feedback2 = 0.0
            else:
                feedback1 = 0.0
                feedback2 = 0.0

            history['fb1_hist'].append(feedback1)
            history['fb2_hist'].append(feedback2)

            #feedback phase
            net.activate([1.0, 0.0, feedback1, feedback2])
        
        fitness = 0.0
        fitness += normal_fitness / (self.cycle * self.cycle_cnt_max * 2)
        fitness += bonus_fitness / (self.E.info['bonus_max'] * 2)

        #history の作成(各サイクルでの正解数・ボーナスタイミングでの正解数が分かるように)
        for idx in range(0, self.cycle_cnt_max * self.cycle, self.cycle):
            history['success_hist'].append(sum(history['fb1_hist'][idx:idx+self.cycle]))
            history['bonus_hist'].append(sum(history['fb2_hist'][idx:idx+self.cycle]))
        #各サイクルで得たfitnessが分かるように
        for idx in range(self.cycle_cnt_max):
            history['fitness_hist'].append(0.0)
            history['fitness_hist'][-1] += history['success_hist'][idx] / (self.cycle * self.cycle_cnt_max * 2)
            history['fitness_hist'][-1] += history['bonus_hist'][idx] / (self.E.info['bonus_max'] * 2)

        return fitness, history

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitness(net)
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Visualize the experiment results
        node_names = {-1:'Bias input', -2: 'observation', -3: 'normal_feedback', -4:'bonus_feedback'}
        net = self.network_type.create(best_genome, config)
        for idx in range(len(net.output_nodes)):
            node_names[idx] = 'output' + str(idx)

        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class multi_cyclic_task:
    def __init__(self, network_type):
        self.network_type = network_type
        self.cycles = [10, 12, 14, 16, 18, 20]
    
    def eval_fitnes(self, net):
        fitness_list, history_list = [], [] #各サブタスクで得たfitness, historyのリスト
        for cycle in self.cycles:
            subtask = single_cyclic_task(network_type = self.network_type, cycle = cycle)
            fitness, history = subtask.eval_fitness(net)
            fitness_list.append(fitness)
            history_list.append(history)

        fitness = sum(fitness_list) / len(fitness_list)

        return fitness, history_list
    
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = self.network_type.create(genome, config)
            genome.fitness, genome.history = self.eval_fitnes(net)
    
    def show_results(self, best_genome, config, stats, out_dir):
        # Visualize the experiment results
        node_names = {-1:'Bias input', -2: 'observation', -3: 'normal_feedback', -4:'bonus_feedback'}
        net = self.network_type.create(best_genome, config)
        for idx in range(len(net.output_nodes)):
            node_names[idx] = 'output' + str(idx)
        visualize.draw_net(config, best_genome, False, node_names=node_names, directory=out_dir)
        visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.png'))
        visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.png'))

class random_cyclic_task:
    def __init__(self,network_type):
        pass

if __name__=='__main__':
    t = single_cyclic_task(0)