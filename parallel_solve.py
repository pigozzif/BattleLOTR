# training settings
import logging
from mpi4py import MPI
import numpy as np
import os
import subprocess
import sys
from es import OpenES
from simulation import battle_simulation
import time


def sprint(*args):
    logging.info(args)  # if python3, can do logging.info()(*args)
    sys.stdout.flush()


class OldSeeder:
    def __init__(self, init_seed=0):
        self._seed = init_seed

    def next_seed(self):
        result = self._seed
        self._seed += 1
        return result

    def next_batch(self, batch_size):
        result = np.arange(self._seed, self._seed + batch_size).tolist()
        self._seed += batch_size
        return result


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2 ** 31 - 1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result


class ParallelExecutor(object):

    def __init__(self, args, num_params, num_episode, eval_steps, num_worker, num_worker_trial, antithetic,
                 cap_time, retrain,
                 seed, sigma_init, sigma_decay):
        self.args = args
        self.num_params = num_params
        self.num_episode = num_episode
        self.num_worker = num_worker
        self.num_worker_trial = num_worker_trial
        self.population = num_worker * num_worker_trial
        self.es = OpenES(num_params,
                         sigma_init=sigma_init,
                         sigma_decay=sigma_decay,
                         sigma_limit=0.02,
                         learning_rate=0.01,
                         learning_rate_decay=1.0,
                         learning_rate_limit=0.01,
                         antithetic=antithetic,
                         weight_decay=0.005,
                         popsize=self.population)
        self.RESULT_PACKET_SIZE = 4 * num_worker_trial
        self.SOLUTION_PACKET_SIZE = (5 + num_params) * num_worker_trial
        self.PRECISION = 10000
        self.seed = seed
        self.antithetic = antithetic
        self.retrain = retrain
        self.eval_steps = eval_steps
        self.cap_time = cap_time

        sprint("process", MPI.COMM_WORLD.Get_rank(), "out of total ", MPI.COMM_WORLD.Get_size(), "started")
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.master()
        else:
            self.slave()

    def master(self):
        start_time = int(time.time())
        sys.stdout.flush()

        seeder = Seeder(self.seed)

        t = 0

        history = []
        eval_log = []
        best_reward_eval = 0
        best_model_params_eval = None

        max_len = -1  # max time steps (-1 means ignore)

        while True:
            t += 1

            solutions = self.es.ask()

            if self.antithetic:
                seeds = seeder.next_batch(int(self.es.popsize / 2))
                seeds = seeds + seeds
            else:
                seeds = seeder.next_batch(self.es.popsize)

            packet_list = self.encode_solution_packets(seeds, solutions, max_len=max_len)

            self.send_packets_to_slaves(packet_list)
            reward_list_total = self.receive_packets_from_slaves()

            reward_list = reward_list_total[:, 0]  # get rewards

            mean_time_step = int(np.mean(reward_list_total[:, 1]) * 100) / 100.  # get average time step
            max_time_step = int(np.max(reward_list_total[:, 1]) * 100) / 100.  # get average time step
            avg_reward = int(np.mean(reward_list) * 100) / 100.  # get average time step
            std_reward = int(np.std(reward_list) * 100) / 100.  # get average time step

            self.es.tell(reward_list)

            r_max = int(np.max(reward_list) * 100) / 100.
            r_min = int(np.min(reward_list) * 100) / 100.

            curr_time = int(time.time()) - start_time

            h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(self.es.rms_stdev() * 100000) / 100000.,
                 mean_time_step + 1.,
                 int(max_time_step) + 1)

            if self.cap_time:
                max_len = 2 * int(mean_time_step + 1.0)
            else:
                max_len = -1

            history.append(h)

            if t == 1:
                best_reward_eval = avg_reward
            if t % self.eval_steps == 0:  # evaluate on actual task at hand

                prev_best_reward_eval = best_reward_eval
                model_params_quantized = np.array(self.es.current_param()).round(4)
                reward_eval = self.evaluate_batch(self.es, model_params_quantized, max_len=-1)
                model_params_quantized = model_params_quantized.tolist()
                improvement = reward_eval - best_reward_eval
                eval_log.append([t, reward_eval, model_params_quantized])
                if len(eval_log) == 1 or reward_eval > best_reward_eval:
                    best_reward_eval = reward_eval
                    best_model_params_eval = model_params_quantized
                else:
                    if self.retrain:
                        sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
                        self.es.set_mu(best_model_params_eval)
                sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best",
                       best_reward_eval)

    def slave(self):
        packet = np.empty(self.SOLUTION_PACKET_SIZE, dtype=np.int32)
        while 1:
            MPI.COMM_WORLD.Recv(packet, source=0)
            assert (len(packet) == self.SOLUTION_PACKET_SIZE)
            solutions = self.decode_solution_packet(packet)
            results = []
            for solution in solutions:
                worker_id, jobidx, seed, train_mode, max_len, weights = solution
                assert (train_mode == 1 or train_mode == 0), str(train_mode)
                worker_id = int(worker_id)
                possible_error = "work_id = " + str(worker_id) + " MPI.COMM_WORLD.Get_rank() = " + str(
                    MPI.COMM_WORLD.Get_rank())
                assert worker_id == MPI.COMM_WORLD.Get_rank(), possible_error
                jobidx = int(jobidx)
                fitness, timesteps = self.worker(weights)
                results.append([worker_id, jobidx, fitness, timesteps])
            result_packet = self.encode_result_packet(results)
            assert len(result_packet) == self.RESULT_PACKET_SIZE
            MPI.COMM_WORLD.Send(result_packet, dest=0)

    def encode_solution_packets(self, seeds, solutions, train_mode=1, max_len=-1):
        n = len(seeds)
        result = []
        for i in range(n):
            worker_num = int(i / self.num_worker_trial) + 1
            result.append([worker_num, i, seeds[i], train_mode, max_len])
            result.append(np.round(np.array(solutions[i]) * self.PRECISION, 0))
        result = np.concatenate(result).astype(np.int32)
        result = np.split(result, self.num_worker)
        return result

    def decode_solution_packet(self, packet):
        packets = np.split(packet, self.num_worker_trial)
        result = []
        for p in packets:
            result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float) / self.PRECISION])
        return result

    def encode_result_packet(self, results):
        r = np.array(results)
        r[:, 2:4] *= self.PRECISION
        return r.flatten().astype(np.int32)

    def decode_result_packet(self, packet):
        r = packet.reshape(self.num_worker_trial, 4)
        workers = r[:, 0].tolist()
        jobs = r[:, 1].tolist()
        fits = r[:, 2].astype(np.float) / self.PRECISION
        fits = fits.tolist()
        times = r[:, 3].astype(np.float) / self.PRECISION
        times = times.tolist()
        result = []
        n = len(jobs)
        for i in range(n):
            result.append([workers[i], jobs[i], fits[i], times[i]])
        return result

    def worker(self, weights, batch_mode=None):
        reward_list, t_list = battle_simulation(self.args, weights)
        if batch_mode == 'min':
            reward = np.min(reward_list)
        else:
            reward = np.mean(reward_list)
        t = np.mean(t_list)
        return reward, t

    def send_packets_to_slaves(self, packet_list):
        num_worker = MPI.COMM_WORLD.Get_size()
        assert len(packet_list) == num_worker - 1
        for i in range(1, num_worker):
            packet = packet_list[i - 1]
            assert (len(packet) == self.SOLUTION_PACKET_SIZE)
            MPI.COMM_WORLD.Send(packet, dest=i)

    def receive_packets_from_slaves(self):
        result_packet = np.empty(self.RESULT_PACKET_SIZE, dtype=np.int32)

        reward_list_total = np.zeros((self.population, 2))

        check_results = np.ones(self.population, dtype=np.int)
        for i in range(1, self.num_worker + 1):
            MPI.COMM_WORLD.Recv(result_packet, source=i)
            results = self.decode_result_packet(result_packet)
            for result in results:
                worker_id = int(result[0])
                possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
                assert worker_id == i, possible_error
                idx = int(result[1])
                reward_list_total[idx, 0] = result[2]
                reward_list_total[idx, 1] = result[3]
                check_results[idx] = 0

        check_sum = check_results.sum()
        assert check_sum == 0, check_sum
        return reward_list_total

    def evaluate_batch(self, es, model_params, max_len=-1):
        # duplicate model_params
        solutions = []
        for i in range(es.popsize):
            solutions.append(np.copy(model_params))

        seeds = np.arange(es.popsize)

        packet_list = self.encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

        self.send_packets_to_slaves(packet_list)
        reward_list_total = self.receive_packets_from_slaves()

        reward_list = reward_list_total[:, 0]  # get rewards
        return np.mean(reward_list)


def mpi_fork(n):
    """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        logging.info(["mpirun", "-np", str(n), sys.executable] + sys.argv)
        subprocess.check_call(["mpiexec", "-n", str(n), sys.executable] + ['-u'] + sys.argv, env=env)
        return "parent"
    else:
        nworkers = MPI.COMM_WORLD.Get_size()
        logging.info('assigning the MPI.COMM_WORLD.Get_rank() and nworkers', nworkers, MPI.COMM_WORLD.Get_rank())
        return "child"
